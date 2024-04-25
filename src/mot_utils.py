#%%
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns

from algs.faceDetect import FaceDetector
from algs.cluster import Clusterer, display_cluster_images
from algs.ReID import encode_folder_crops_to_reid, load_ReID_model
from metric.similarity import cosine_similarity, get_dist_mat

import warnings
warnings.filterwarnings("ignore")


def crop_targets_from_video(video_path, crops, output_dir=None):
    """
    Crops and optionally saves images from a video based on provided bounding boxes.

    Parameters:
        video_path (str): The path to the video file.
        crops (list of tuples): A list of tuples, each containing (frame_id, x, y, width, height).
        output_dir (str, optional): Directory to save the cropped images. If None, images are not saved.

    Returns:
        list of numpy arrays: The cropped images.
    """
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open the video file")

    cropped_images = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_fns = []
    for crop in crops:
        frame_id = crop['frame_id']
        x1, y1, w, h = tlwh = crop['tlwh']
        
        if frame_id >= frame_count:
            continue

        # Set the video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        crop_img = frame[max(0, y1): h, max(0, x1): w]

        if output_dir:
            img_fn = os.path.join(output_dir, f"{crop['track_id']:03d}_{frame_id:04d}.jpg")
            img_fns.append(img_fn)
            cv2.imwrite(img_fn, crop_img)
        
        cropped_images.append(crop_img)

    cap.release()
    
    return cropped_images, img_fns

def load_mot_result(video_name = "IMG_2230", crop_folder=None):
    video_path = f"../data/videos/{video_name}.MOV"
    mot_res_path = f"../data/mot/{video_name}.csv"

    df = pd.read_csv(mot_res_path)
    df = df[df.frame_id % 10 == 0]
    
    df.tlwh = df.tlwh.map(eval)
    df.feature = df.feature.apply(eval)
    df.loc[:, 'video'] = video_name

    if crop_folder:
        cropped_images, fns = crop_targets_from_video(
            video_path, 
            df.to_dict(orient='records'), 
            output_dir = crop_folder + f'/{video_name}'
        )

        df['crop_fn'] = fns
        
    return df

def detect_and_idntify_faces(recognizer, fns):
    res = []
    for fn in fns:
        crop = cv2.imread(fn)
        res.append(recognizer.detect_and_identify(crop))

    df_face = pd.json_normalize(res).fillna(np.nan)
    
    return df_face


# %%
# 获取 crops 
videos = ['IMG_2230', 'IMG_2231', 'IMG_2232']

res = []
for video in videos:
    df = load_mot_result(video_name = video, crop_folder="../data/crops")
    res.append(df)
    
df_all = pd.concat(res)
df_all.head()


#%%

recognizer = FaceDetector(gallery_path='../data/gallery')
df_face = detect_and_idntify_faces(recognizer, df_all.crop_fn)


#%%

reid_model = load_ReID_model(model_dir='../ckpt/reid_model')
_, appearance_gallery = encode_folder_crops_to_reid(reid_model, '../data/app_gallery/wenke', is_normed=True)


#%%
video_name = videos[0]
track_id = 1
df = df_all.query("video == @video_name and track_id == @track_id")
df

#%%

feats = np.vstack(df.feature)
dis_mat = np.abs(1 - cosine_similarity(feats, feats))

# %%
cluster = Clusterer('hdbscan')
labels = cluster.fit_predict(dis_mat)
sorted_clusters = cluster.sort_cluster_members(dis_mat, labels)
sorted_clusters


# %%
display_cluster_images(sorted_clusters, df.crop_fn, n=5)

# %%

appearance_sim_mat = cosine_similarity(feats, appearance_gallery)

# %%
