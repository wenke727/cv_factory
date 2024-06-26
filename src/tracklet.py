#%%
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from datetime import datetime
from collections import defaultdict

from algs.searcher import GallerySearcher
from algs.cluster import Clusterer, display_cluster_images
from metric.similarity import cosine_similarity
from gallery import load_and_unpack_gallery_data

MINIMUN_SIMILARITY_THRED = 0.3

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

def get_top_k_mean(arr, k=5):
    if len(arr) == 0:
        return 0

    return np.sum(sorted(arr, reverse=True)[:k]) / k

class MultiModalFusion:
    def __init__(self, face_weight = 0.5, body_weight = 0.25, voice_weight = 0.25):
        self.weights = {'face': face_weight, 'body': body_weight, 'voice': voice_weight}

        self.score_history = defaultdict(list)

        self.face_history = defaultdict(list)
        self.body_history = defaultdict(list)
        self.voice_history = defaultdict(list)

    def add_score(self,
                  face_uuid:str = None,
                  body_uuid:str = None,
                  face_score:float = None,
                  body_score:float = None,
                  voice_score:float = None,
                  face_fusion_threshold:float = MINIMUN_SIMILARITY_THRED):
        """
        Add a new score to the history based on the weighted sum of individual modality scores,
        considering a threshold for the inclusion of face scores.

        Parameters:
            face_score (float, optional): The confidence score from face recognition.
            body_score (float, optional): The confidence score from body recognition.
            voice_score (float, optional): The confidence score from voice recognition.

        Returns:
            None. Updates the internal score history with the calculated weighted score.
        """
        if face_uuid is not None and face_uuid != body_uuid:
            if face_score * 2 >= body_score:
                logger.warning(f"😊 Mismatch in UUIDs - Face: {face_uuid}, Body: {body_uuid}. Prioritizing based on scores.")
                # TODO 容易混淆的外观特征
                body_score = None
                body_uuid = None
            else:
                logger.warning(f"🚶 Mismatch in UUIDs - Face: {face_uuid}, Body: {body_uuid}. Prioritizing based on scores.")
                face_score = None
                face_uuid = None

            body_score = 0
            body_uuid = None

        if face_uuid is not None:
            uuid = face_uuid
        elif body_uuid is not None:
            uuid = body_uuid
        else:
            return {'conf': 0}

        weight = 0
        score = 0
        if face_score is not None and face_score > face_fusion_threshold:
            score += face_score * self.weights['face']
            weight += self.weights['face']
            self.face_history[uuid].append(face_score)
        if body_score is not None and body_score > 0:
            score += body_score * self.weights['body']
            weight += self.weights['body']
            self.body_history[uuid].append(body_score)
        if voice_score is not None:
            score += voice_score * self.weights['voice']
            weight += self.weights['voice']
            self.voice_history[uuid].append(voice_score)
        if weight > 0:
            score = score / weight

        self.score_history[uuid].append(score)

        # TODO 若是这个 uuid 不是最大的，则还是返回至大至，可适当考虑长度系数
        return self.get_fusion_score(uuid)

    def get_fusion_score(self, uuid, k=5):
        """
        Calculate the average of the square roots of the top k scores from the score history.

        Parameters:
            k (int): The number of top scores to consider for calculating the average.

        Returns:
            float: The average of the square roots of the top k scores. Returns None if there are
                   fewer than k scores available in the history.

        This method enhances the impact of higher scores on the final result by computing the
        square root of the scores, which mitigates the influence of outlier low scores and
        emphasizes higher values more than a simple average would.
        """
        if len(self.score_history.get(uuid, [])) < k:
            return {'conf': 0, 'status': 1}

        uuids = self.score_history.keys()
        uuid_2_score = {}
        for id in uuids:
            uuid_2_score[id]= get_top_k_mean(self.score_history.get(id, []), k)

        if uuid_2_score.get(uuid, 0) < max(uuid_2_score.values()):
            logger.warning(f"和历史有冲突，{uuid_2_score}")
            return {'conf': 0, 'status': 2}

        scores = {}
        if self.score_history[uuid]:
            scores['conf'] = uuid_2_score[uuid]
        if self.face_history[uuid]:
            scores['face'] = get_top_k_mean(self.face_history[uuid], k)
        if self.body_history[uuid]:
            scores['body'] = get_top_k_mean(self.body_history[uuid], k)
        if self.voice_history[uuid]:
            scores['voice'] = get_top_k_mean(self.voice_history[uuid], k)

        return scores

    def is_confirmed(self, threshold=.5):
        average_score = self.get_fusion_score()

        if average_score is not None and average_score > threshold:
            return True

        return False

    def update_weights(self, light_condition, noise_level):
        # Change the wieghts by surroundings. But the ability to sense the env need to be build.
        if light_condition < 0.5:
            self.weights['face'] -= 0.2
            self.weights['body'] += 0.1
            self.weights['voice'] += 0.1
        if noise_level > 0.5:
            self.weights['voice'] -= 0.1
            self.weights['face'] += 0.05
            self.weights['body'] += 0.05

class Tracklet:
    def __init__(self,
                 track_id,
                 face_seacher:GallerySearcher,
                 appearance_searcher:GallerySearcher,
                 voice_searcher:GallerySearcher = None):
        self.track_id = track_id
        self.timestamps = []
        self.crop_imgs = {}
        self.crop_fns = []

        # reid
        self.body_embs = []
        self.body_tlwhs = []
        self.face_embs = []
        self.face_tlwhs = []

        # others attrs
        self.data = defaultdict(dict)
        self.last_seen = None

        # gallery seacher
        self.face_seacher = face_seacher
        self.appearance_searcher = appearance_searcher
        self.voice_searcher = voice_searcher

        self.fusion_modle = MultiModalFusion(face_weight=.5, body_weight=.25, voice_weight=.25)
        # self.state = 'active'  # Options: active, inactive, completed

    def update(self, timestamp, face_emb, body_emb, crop=None, atts: dict={}):
        """
        Update the tracking information for a frame.

        Parameters:
            timestamp (datetime.datetime): The timestamp of the update.
            face_emb (np.ndarray): Embedding for the face detected.
            body_emb (np.ndarray): Embedding for the body detected.
            crop (Optional[Any]): Cropped image of the detected entity.
            atts (Dict[str, Any]): Additional attributes to record.
        """
        self.timestamps.append(timestamp)
        self.face_embs.append(np.array(face_emb) if not np.isnan(face_emb).all() else np.nan)
        self.body_embs.append(np.array(body_emb) if not np.isnan(body_emb).all() else np.nan)
        for k, v in atts.items():
            self.data[k][timestamp] = v
        if crop is not None:
            self.crop_imgs[timestamp] = crop
        if 'crop_fn' in atts:
            self.crop_fns.append(atts['crop_fn'])

        self.last_seen = timestamp
        self.identify()

    def identify(self, similarity_thres=.3, precise=4, verbose=True):
        """
        Identifies and fuses modalities per frame using dynamic weights based on environmental
        factors and modality reliability, specifically adjusting face recognition contributions.

        This function calculates similarity scores for face and body modalities, applies a threshold to
        filter out low-confidence results, and then dynamically fuses these scores to determine a combined result.
        It prefers face recognition results but considers body recognition when face data is not conclusive.

        Args:
            similarity_thres (float): The threshold for recognition confidence.
            precise (int): The number of decimal places to which results should be rounded.
            verbose (bool): If True, logs detailed debug information.

        Returns:
            dict: Contains 'face_score', 'body_score', and a decision based on the fused result.

        Design Principle of similarity_thres:
            The similarity_thres parameter is designed to provide a control mechanism that filters out
            low-confidence face recognition results, preventing them from influencing the overall identity
            decision. This is particularly useful in environments where face data can be highly variable, or
            when other modalities (like body or voice data) need to compensate for uncertainties in face data.
        """
        face_score, face_uuid = self.search_by_face(similarity_thres)
        body_score, body_uuid = self.search_by_appearance(similarity_thres)

        # deal with the face and appearane conflit
        similarity = {}
        if face_score is not None:
            similarity['face_score'] = face_score
        if body_score is not None:
            similarity['body_score'] = body_score

        score_fusion = self.fusing(face_uuid, body_uuid, **similarity)
        score_fusion = {k : round(v, precise)for k, v in score_fusion.items()}

        if verbose:
            info = f"{self.last_seen:3d}, Similarity: "
            if face_uuid is not None:
                info += f"😊 {face_uuid}: {similarity.get('face_score'):.4f}, "
            info += f"🚶 {body_uuid}: {similarity.get('body_score'):.4f}, "
            if score_fusion.get('conf'):
                info += f"🚀 conf: {score_fusion.get('conf')}"
            logger.debug(info)

        return similarity

    def search_by_face(self, similarity_thres=.3):
        if len(self.face_embs) == 0:
            return 0, None

        score, _, uuid = self.face_seacher.search_best(self.face_embs[-1], similarity_thres)

        return score, uuid

    def search_by_appearance(self, similarity_thres=.3):
        if len(self.body_embs) == 0:
            return 0, None

        score, _, uuid = self.appearance_searcher.search_best(self.body_embs[-1], similarity_thres)

        return score, uuid

    def fusing(self, face_uuid, body_uuid, face_score=None, body_score=None, voice_score=None):
        return self.fusion_modle.add_score(face_uuid, body_uuid, face_score, body_score, voice_score)

    def is_active(self, timeout_seconds=300):
        return (datetime.datetime.now() - self.last_seen).total_seconds() < timeout_seconds

    def distill_feat(self, plot=False):
        # Assuming all body encodings are stored in a list
        body_feats = np.vstack(self.body_embs)
        dis_mat = np.abs(1 - cosine_similarity(body_feats, body_feats))

        cluster = Clusterer('hdbscan')
        labels = cluster.fit_predict(dis_mat)
        sorted_clusters = cluster.sort_cluster_members(dis_mat, labels)

        if plot:
            fns = pd.Series(self.crop_fns, index=self.timestamps)
            display_cluster_images(sorted_clusters, fns, n = 6)

        rep_feats = []
        for k, vals in sorted_clusters.items():
            rep_feats.append([vals[0], tracker.body_embs[vals[0]]])

        return rep_feats, sorted_clusters

    def get_latest(self):
        if len(self.timestamps) == 0:
            return None

        return {
            'encoding': self.face_embs[-1],
            'location': self.face_locations[-1],
            'timestamp': self.timestamps[-1]
        }

    def convert_2_dataframe(self, with_atts=False):
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'body_emb': self.body_embs,
            'face_emb': self.face_embs
        }).set_index('timestamp')

        if with_atts:
            df = pd.concat([df, pd.DataFrame.from_dict(self.data)], axis=1)

        return df

    def __repr__(self):
        return f"<Tracklet {self.track_id} | Last seen: {self.last_seen} >" # | State: {self.state}


if __name__ == "__main__":
    # 1. prepare tracklet
    tracks = convert_hdf_to_tracklet_data(fn='../data/feats/IMG_2232.h5')
    track_id = 1
    track_data = tracks[track_id]

    # 2. prepare gallery
    face_gallery, face_index_to_user, appearance_gallery, appearance_index_to_user = \
        load_and_unpack_gallery_data('../data/gallery/gallery.ckpt')

    face_searcher = GallerySearcher(feat_len=512)
    face_searcher.add(face_gallery, face_index_to_user)

    appearance_searcher = GallerySearcher(feat_len=256)
    appearance_searcher.add(appearance_gallery, appearance_index_to_user)

    track_data[85] = {
        'face_emb': face_gallery[2], # zhaoming
        'body_emb': track_data[100]['body_emb'], # zhaoming
    }
    sorted_keys = sorted(track_data)
    track_data = {key: track_data[key] for key in sorted_keys}
    track_data.keys()

    # 3. main
    tracker = Tracklet(track_id, face_searcher, appearance_searcher)

    for t, atts in track_data.items():
        tracker.update(timestamp = t, **atts)

    uuids = tracker.fusion_modle.score_history.keys()
    logger.info(f"uuids: {uuids}")


    logger.info(f"Face, wenke: {tracker.fusion_modle.face_history['wenke']}")
    logger.info(f"Face, zhaoming: {tracker.fusion_modle.face_history['zhaoming']}")

    logger.info(f"Body, wenke: {tracker.fusion_modle.body_history['wenke']}")
    logger.info(f"Body, zhaoming: {tracker.fusion_modle.body_history['zhaoming']}")

    logger.info(f"Fusion, wenke: {tracker.fusion_modle.get_fusion_score('wenke')}")
    logger.info(f"Fusion, zhaoming: {tracker.fusion_modle.get_fusion_score('zhaoming')}")


    #%%
    df = tracker.convert_2_dataframe(with_atts=True)
    df

    #%%
    # distill
    rep_feats, _ = tracker.distill_feat(plot=True)


# %%



# %%
