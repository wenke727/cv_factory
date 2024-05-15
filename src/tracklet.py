#%%
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from loguru import logger
from datetime import datetime
from sklearn.cluster import KMeans
from collections import defaultdict

from algs.faceDetect import FaceDetector
from algs.cluster import Clusterer, display_cluster_images
from metric.similarity import cosine_similarity, get_dist_mat
from utils_helper.serialization import load_checkpoint


def convert_hdf_to_tracklet_data(fn):
    df_feats = pd.read_hdf(fn)
    trackid_2_tracks = {}

    for track_id in df_feats.track_id.unique():
        track_id = 1
        df_feats.query("track_id == @track_id", inplace=True)
        df_feats.drop(columns=['track_id'], inplace=True)
        df_feats.rename(columns={'frame_id': 'timestamp'}, inplace=True)
        records = df_feats.to_dict(orient='records')

        timestamp_2_data = {}
        for item in records:
            item = item.copy()

            timestamp = item['timestamp']
            body_emb = item['reid_emb']
            face_emb = item['face_emb']
            del item['timestamp']
            del item['reid_emb']
            del item['face_emb']

            timestamp_2_data[timestamp] = {
                "face_emb": face_emb,
                "body_emb": body_emb,
                "atts": item
            }

        trackid_2_tracks[track_id] = timestamp_2_data

    return trackid_2_tracks

user_gallery = load_checkpoint('../data/gallery/wenke/gallery.ckpt')
user_gallery['face_gallery'] = user_gallery['face_gallery'][np.newaxis, :]
logger.info(f"appearance_filenames: {user_gallery['appearance_filenames']}")
user_gallery

tracks = convert_hdf_to_tracklet_data(fn='../data/feats/IMG_2232.h5')

track_id = 1
track_data = tracks[track_id]
# track_data

#%%

class Tracklet:
    def __init__(self, track_id):
        self.track_id = track_id
        self.timestamps = []
        self.crop_imgs = {} # cv2::img

        self.face_gallery = np.array([])
        self.voice_fallery = np.array([])
        self.appearance_gallery = np.array([])

        # reid
        self.body_embs = []
        self.face_embs = []
        # self.face_embs = []

        # others attrs
        self.data = defaultdict(dict)

        self.match_history = []
        self.last_seen = None
        # self.state = 'active'  # Options: active, inactive, completed

    def update(self, timestamp, face_emb, body_emb, crop=None, atts: dict={}):
        """_summary_

        Args:
            timestamp (_type_): _description_
            face_emb (_type_): _description_
            body_emb (_type_): _description_
            crop (_type_, optional): _description_. Defaults to cv2 image.
            atts (dict, optional): _description_. Defaults to {}.
        """
        self.timestamps.append(timestamp)
        self.face_embs.append(np.array(face_emb) if not np.isnan(face_emb).all() else np.nan)
        self.body_embs.append(np.array(body_emb) if not np.isnan(body_emb).all() else np.nan)
        for k, v in atts.items():
            self.data[k][timestamp] = v
        if crop is not None:
            self.crop_imgs[timestamp] = crop

        self.last_seen = timestamp

        # TODO Here, you would also want to handle state updates and matching logic

    def identify_per_frame(self, verbose=True, face_fusion_thred=.4):
        similarity = {}
        if self.appearance_gallery.size > 0:
            body_sim = cosine_similarity(self.body_embs[-1][np.newaxis, :], self.appearance_gallery)[0]
            similarity['body'] = round(body_sim.max(), 3)

        if self.face_gallery.size > 0 and isinstance(self.face_embs[-1], np.ndarray):
            face_sim = cosine_similarity(self.face_embs[-1][np.newaxis, :], self.face_gallery)[0][0]
            similarity['face'] = round(face_sim, 3)

        # Determine the weights for each modality
        weights = {'body': 0.25, 'face': 0.5}  # Adjust these based on empirical evidence or domain knowledge

        # Calculate the weighted average similarity
        if similarity and similarity.get('face', 0) > face_fusion_thred:
            final_similarity = sum(similarity[mod] * weights[mod] for mod in similarity) \
                / sum(weights[mod] for mod in similarity)
            similarity['final_similarity'] = round(final_similarity, 3)

        if verbose:
            logger.debug(f"timestamp: {self.last_seen:4d}, similary: {str(similarity)}")

        return 0

    def identify(self):
        # Implement logic to consolidate identity across the tracked frames
        pass

    def set_gallery(self, face_gallery, appearance_gallery, voice_gallery=None):
        self.face_gallery = face_gallery
        self.appearance_gallery = appearance_gallery
        self.voice_fallery = voice_gallery

    def is_active(self, timeout_seconds=300):
        return (datetime.datetime.now() - self.last_seen).total_seconds() < timeout_seconds

    def calculate_similarity(self, encoding):
        # Assuming the use of cosine similarity
        latest_encoding = self.face_embs[-1]
        cos_sim = np.dot(latest_encoding, encoding) / (np.linalg.norm(latest_encoding) * np.linalg.norm(encoding))

        return cos_sim

    def distill_feat(self, plot=False):
        # Assuming all body encodings are stored in a list
        body_feats = np.vstack(self.body_embs)
        dis_mat = np.abs(1 - cosine_similarity(body_feats, body_feats))

        cluster = Clusterer('hdbscan')
        labels = cluster.fit_predict(dis_mat)
        sorted_clusters = cluster.sort_cluster_members(dis_mat, labels)

        if plot:
            display_cluster_images(sorted_clusters, df.crop_fn, n=6)

        rep_feats = []
        for k, vals in sorted_clusters.items():
            rep_feats.append([vals[0], tracker.body_embs[vals[0]]])

        return rep_feats, sorted_clusters

    def get_latest(self):
        return {
            'encoding': self.face_embs[-1],
            'location': self.face_locations[-1],
            'timestamp': self.timestamps[-1]
        }

    def __repr__(self):
        return f"<Tracklet {self.track_id} | Last seen: {self.last_seen} >" # | State: {self.state}

    def convert_2_dataframe(self, with_atts=False):
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'body_emb': self.body_embs,
            'face_emb': self.face_embs
        }).set_index('timestamp')

        if with_atts:
            df = pd.concat([df, pd.DataFrame.from_dict(self.data)], axis=1)

        return df


# if __name__ == "__main__":

tracker = Tracklet(track_id)
tracker.set_gallery(
    face_gallery = user_gallery['face_gallery'],
    appearance_gallery = user_gallery['appearance_gallery'])

for t, atts in track_data.items():
    tracker.update(timestamp = t, **atts)
    tracker.identify_per_frame()

# distill
# rep_feats, _ = tracker.distill_feat(plot=True)

df = tracker.convert_2_dataframe(with_atts=True)


# %%
