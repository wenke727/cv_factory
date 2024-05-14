#%%
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from collections import defaultdict

from algs.faceDetect import FaceDetector
from algs.cluster import Clusterer, display_cluster_images
from algs.ReID import encode_folder_crops_to_reid, load_ReID_model
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

user_gallery

tracks = convert_hdf_to_tracklet_data(fn='../data/feats/IMG_2232.h5')

track_id = 1
track_data = tracks[track_id]
track_data

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
        self.face_embs.append(face_emb)
        self.body_embs.append(body_emb)
        for k, v in atts.items():
            self.data[k][timestamp] = v
        if crop is not None:
            self.crop_imgs[timestamp] = crop

        self.last_seen = timestamp

        # TODO Here, you would also want to handle state updates and matching logic

    def identify_per_frame(self):
        # Implement logic to identify individual based on current frame's data
        # This should include multimodal data fusion
        pass

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

    def distill_feat(self):
        # Assuming all body encodings are stored in a list
        all_encodings = np.vstack(self.body_embs)
        kmeans = KMeans(n_clusters=1)  # You might want to set the number of clusters dynamically
        kmeans.fit(all_encodings)
        updated_feature = kmeans.cluster_centers_[0]
        return updated_feature

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

df = tracker.convert_2_dataframe(with_atts=True)
df

# %%
body_feats = np.vstack(df.body_emb)
dis_mat = np.abs(1 - cosine_similarity(body_feats, body_feats))

cluster = Clusterer('hdbscan')
labels = cluster.fit_predict(dis_mat)
sorted_clusters = cluster.sort_cluster_members(dis_mat, labels)
display_cluster_images(sorted_clusters, df.crop_fn, n=6)

sorted_clusters

# %%
body_feats = np.vstack(df.body_emb)
dis_mat = np.abs(1 - cosine_similarity(body_feats, user_gallery['appearance_gallery']))

sns.heatmap(dis_mat, annot=True)


# %%
body_feats = np.vstack(df.face_emb.dropna())
dis_mat = np.abs(1 - cosine_similarity(body_feats, user_gallery['face_gallery']))

sns.heatmap(dis_mat, annot=True)


# %%
user_gallery['face_gallery'].unsqueeze().shape


# %%
