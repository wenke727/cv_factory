#%%
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from collections import defaultdict

from algs.faceDetect import FaceDetector
from algs.cluster import Clusterer, display_cluster_images
from algs.ReID import encode_folder_crops_to_reid, load_ReID_model
from metric.similarity import cosine_similarity, get_dist_mat

# 加载人脸识别模型
# recognizer = FaceDetector(gallery_path='../data/gallery')
# reid_model = load_ReID_model(model_dir='../ckpt/reid_model')

# _, appearance_gallery = encode_folder_crops_to_reid(reid_model, '../data/app_gallery/wenke', is_normed=True)

df_feats = pd.read_hdf('../data/feats/IMG_2232.h5')
track_id = 1
df_feats.query("track_id == @track_id", inplace=True)
df_feats.drop(columns=['track_id'], inplace=True)
df_feats.rename(columns={'frame_id': 'timestamp'}, inplace=True)
records = df_feats.to_dict(orient='records')
df_feats

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
    
timestamp_2_data


#%%

class Tracklet:
    def __init__(self, track_id):
        self.track_id = track_id
        self.timestamps = []
        self.crop_imgs = [] # cv2::img

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

    def identify_per_frame(self):
        # Implement logic to identify individual based on current frame's data
        # This should include multimodal data fusion
        pass

    def identify(self):
        # Implement logic to consolidate identity across the tracked frames
        pass

    def set_gallery(self, face_gallery, voice_gallery, appearance_gallery):
        self.face_gallery = face_gallery
        self.appearance_gallery = appearance_gallery
        self.voice_fallery = voice_gallery

    def is_active(self, timeout_seconds=300):
        return (datetime.datetime.now() - self.last_seen).total_seconds() < timeout_seconds

    def update(self, timestamp, face_emb, body_emb, atts:dict):
        self.timestamps.append(timestamp)
        self.face_embs.append(face_emb)
        self.body_embs.append(body_emb)
        self.last_seen = timestamp
        
        for k, v in atts.items():
            self.data[k][timestamp] = v
        
        # Here, you would also want to handle state updates and matching logic

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


# if __name__ == "__main__":
tracker = Tracklet(track_id)

for t, atts in timestamp_2_data.items():
    tracker.update(timestamp = t, **atts)

tracker

# %%
len(tracker.body_embs)

# %%
pd.DataFrame.from_dict(tracker.data)

# %%
