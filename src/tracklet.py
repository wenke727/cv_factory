#%%
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from datetime import datetime
from collections import defaultdict

from algs.cluster import Clusterer, display_cluster_images
from metric.similarity import cosine_similarity
from gallery import load_and_unpack_gallery_data

"""
TODO
x 1. 家庭的 gallery
2. face 和 body 冲突的情况
"""

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

class MultiModalFusion:
    def __init__(self, face_weight=0.5, body_weight=0.25, voice_weight=0.25):
        self.weights = {'face': face_weight, 'body': body_weight, 'voice': voice_weight}
        self.score_history = []
        self.face_history = []
        self.body_history = []
        self.voice_history = []

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

    def add_score(self,
                  face_score:float = None,
                  body_score:float = None,
                  voice_score:float = None,
                  face_fusion_threshold: float=0.4):
        """
        Add a new score to the history based on the weighted sum of individual modality scores,
        considering a threshold for the inclusion of face scores.

        Parameters:
            face_score (float, optional): The confidence score from face recognition.
            body_score (float, optional): The confidence score from body recognition.
            voice_score (float, optional): The confidence score from voice recognition.
            face_fusion_threshold (float): The minimum threshold for face recognition score
                                           to be considered in the fusion calculation.
                                           Scores below this value are disregarded for the face modality.

        Returns:
            None. Updates the internal score history with the calculated weighted score.

        The function calculates a combined score by applying predefined weights to the scores of
        individual modalities. If the face score is below a specified threshold, it is not included
        in the final score calculation. This method allows dynamic adjustment of the influence of
        each modality based on its performance and reliability.
        """
        weight = 0
        score = 0
        if face_score is not None and face_score > face_fusion_threshold:
            score += face_score * self.weights['face']
            weight += self.weights['face']
            self.face_history.append(face_score)

        if body_score is not None:
            score += body_score * self.weights['body']
            weight += self.weights['body']
            self.body_history.append(body_score)

        if voice_score is not None:
            score += voice_score * self.weights['voice']
            weight += self.weights['voice']
            self.voice_history.append(voice_score)

        if weight > 0:
            score = score / weight

        self.score_history.append(score)

        return self.get_fusion_score()

    def get_fusion_score(self, k=5):
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
        if len(self.score_history) < k:
            return {}

        def get_top_k_mean(atts):
            return np.mean(sorted(atts, reverse=True)[:k])

        scores = {}
        if self.score_history:
            scores['conf'] = get_top_k_mean(self.score_history)
        if self.face_history:
            scores['face'] = get_top_k_mean(self.face_history)
        if self.body_history:
            scores['body'] = get_top_k_mean(self.body_history)
        if self.voice_history:
            scores['voice'] = get_top_k_mean(self.voice_history)

        return scores

    def is_confirmed(self, threshold=.6):
        average_score = self.get_fusion_score()

        if average_score is not None and average_score > threshold:
            return True

        return False

class Tracklet:
    def __init__(self, track_id):
        self.track_id = track_id
        self.timestamps = []
        self.crop_imgs = {} # cv2::img
        self.crop_fns = []

        self.face_gallery = np.array([])
        self.voice_fallery = np.array([])
        self.appearance_gallery = np.array([])

        # reid
        self.body_embs = []
        self.body_tlwhs = []
        self.face_embs = []
        self.face_tlwhs = []

        # others attrs
        self.data = defaultdict(dict)

        self.match_history = []
        self.last_seen = None

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

        The function calculates similarity scores for face and body modalities, applies a threshold to
        filter out low-confidence results, and then fuses these scores to determine a combined result.

        Args:
            similarity_thres (float): The threshold for recognition confidence. Scores below this
                                    threshold are not considered reliable and are thus filtered out.
                                    Default is 0.3.
            precise (int): The number of decimal places to which results should be rounded. This affects
                        the precision of the output scores. Default is 4.
            verbose (bool): If set to True, detailed debug information, including candidate indices and scores,
                            will be logged. Default is True.

        Returns:
            dict: A dictionary containing 'face_score' and 'body_score' if they exceed the similarity threshold.
                The function may be modified to return additional information or decisions based on these scores.

        Design Principle of similarity_thres:
            The similarity_thres parameter is designed to provide a control mechanism that filters out
            low-confidence face recognition results, preventing them from influencing the overall identity
            decision. This is particularly useful in environments where face data can be highly variable, or
            when other modalities (like body or voice data) need to compensate for uncertainties in face data.
        """
        cand_face_idx, face_score = self.calculate_similarity(
            self.face_embs, self.face_gallery, similarity_thres, precise)
        cand_body_idx, body_score = self.calculate_similarity(
            self.body_embs, self.appearance_gallery, similarity_thres,precise)
        # TODO cand_face_idx 和 cand_body_idx 打架的情况

        similarity = {}
        if face_score is not None:
            similarity['face_score'] = face_score
        if body_score is not None:
            similarity['body_score'] = body_score

        score_fusion = self.add_score(**similarity)
        score_fusion = {k : round(v, precise)for k, v in score_fusion.items()}

        if verbose:
            info = "cands, "
            if cand_face_idx is not None:
                info += f"face {cand_face_idx}: {similarity.get('face_score'):.4f}, "
            info += f"appearcne {cand_body_idx}: {similarity.get('body_score')}, "
            info += f"conf: {score_fusion.get('conf')}"
            logger.debug(info)

        return similarity

    def add_score(self, face_score=None, body_score=None, voice_score=None, face_fusion_threshold=0.4):
        return self.fusion_modle.add_score(face_score, body_score, voice_score, face_fusion_threshold)

    def set_gallery(self, face_gallery, appearance_gallery, voice_gallery=None):
        self.face_gallery = face_gallery
        self.appearance_gallery = appearance_gallery
        self.voice_fallery = voice_gallery

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

    def calculate_similarity(self, embeddings, gallery, similarity_thres=.3, precise=4):
        if gallery.size > 0 and isinstance(embeddings[-1], np.ndarray):
            emb = embeddings[-1][np.newaxis, :]
            sim = cosine_similarity(emb, gallery)[0]
            cand_idx = sim.argmax()

            if sim[cand_idx] > similarity_thres:
                return cand_idx, round(sim[cand_idx], precise)

        return None, 0


if __name__ == "__main__":
    face_gallery, face_index_to_user, appearance_gallery, appearance_index_to_user = \
        load_and_unpack_gallery_data('../data/gallery/gallery.ckpt')

    tracks = convert_hdf_to_tracklet_data(fn='../data/feats/IMG_2232.h5')
    track_id = 1
    track_data = tracks[track_id]

    tracker = Tracklet(track_id)
    tracker.set_gallery(face_gallery, appearance_gallery)

    for t, atts in track_data.items():
        tracker.update(timestamp = t, **atts)

    #%%
    df = tracker.convert_2_dataframe(with_atts=True)
    df

    #%%
    # distill
    rep_feats, _ = tracker.distill_feat(plot=True)


# %%
