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

# 加载人脸识别模型
recognizer = FaceDetector(gallery_path='../data/gallery')
reid_model = load_ReID_model(model_dir='../ckpt/reid_model')

_, appearance_gallery = encode_folder_crops_to_reid(reid_model, '../data/app_gallery/wenke', is_normed=True)

#%%
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
        x1, y1, x2, y2 = bbox = crop['bbox']

        if frame_id >= frame_count:
            continue

        # Set the video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        crop_img = frame[max(0, y1): y2, max(0, x1): x2]

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

    df['bbox'] = df.bbox.map(eval)
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

def detect_and_idntify_faces(recognizer: FaceDetector, fns, identify=False, match_threshold=.6):
    res = []
    for fn in fns:
        crop = cv2.imread(fn)
        if identify:
            tmp = recognizer.detect_and_identify(crop, match_threshold=match_threshold)
        else:
            tmp = recognizer.detect(crop)

        res.append(tmp)

    df_face = pd.json_normalize(res).fillna(np.nan)

    return df_face

def multimodal_similarity(face_feat1, body_feat1, voice_feat1, face_feat2, body_feat2, voice_feat2, weights=(0.4, 0.4, 0.2)):
    """计算多模态相似度"""
    face_sim = cosine_similarity(face_feat1, face_feat2)
    body_sim = cosine_similarity(body_feat1, body_feat2)
    voice_sim = cosine_similarity(voice_feat1, voice_feat2)

    # 加权平均
    total_similarity = face_sim * weights[0] + body_sim * weights[1] + voice_sim * weights[2]

    return total_similarity


# %%
# 获取 crops
videos = ['IMG_2230', 'IMG_2231', 'IMG_2232', 'IMG_0169', "IMG_0174"]
videos = ["IMG_2232"]

res = []
for video in videos:
    df = load_mot_result(video_name = video, crop_folder="../data/crops")
    res.append(df)

df_all = pd.concat(res).reset_index(drop=True)
df_all


#%%
df_face = detect_and_idntify_faces(recognizer, df_all.crop_fn, identify=False)
df_face

#%%
face_renmame_dict = {
    'tlwh': 'face_tlwh',
    'embedding': 'face_emb',
    'confidence': 'face_conf',
    'similarity': 'face_sim',
    'username': 'face_uuid',
}

reid_rename_dict = {
    'conf': 'reid_conf',
    'bbox': 'reid_tlwh',
    'feature': 'reid_emb',
}

feats = pd.concat([
    df_all.rename(columns=reid_rename_dict),
    df_face.rename(columns=face_renmame_dict)], axis=1)

#%%
attrs = [
    # 'video',
    'crop_fn',
    # 'quality',
    # 'reid_tlwh',

    # track 信息
    'track_id',
    'frame_id',

    # reid 信息
    'reid_emb',
    # 'reid_conf',

    # 人脸信息，出现多张脸或者没有人脸的情况下，以下三个值为 None
    'face_tlwh',
    'face_emb',
    'face_conf',

    # 通过人脸匹配的用户
    # 'face_sim',
    # 'face_uuid',
]

feats[attrs].to_hdf(f"../data/feats/{video}.h5", 'feats')



#%%
# video_name = videos[-1]
# track_id = 1
# df = df_all.query("video == @video_name and track_id == @track_id")
# df

#%%
feats = np.vstack(df.feature)
dis_mat = np.abs(1 - cosine_similarity(feats, feats))

cluster = Clusterer('hdbscan')
labels = cluster.fit_predict(dis_mat)
sorted_clusters = cluster.sort_cluster_members(dis_mat, labels)
display_cluster_images(sorted_clusters, df.crop_fn, n=9)

sorted_clusters

# %%
appearance_sim_mat = cosine_similarity(feats, appearance_gallery)
appearance_sim_mat

# %%

