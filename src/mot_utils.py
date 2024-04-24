#%%
import os
import cv2
import numpy as np
import pandas as pd
from loguru import logger

import warnings
warnings.filterwarnings("ignore")

""" Byte track """
def correct_tlwh(tlwh, image_width=np.float('inf'), image_height=np.float('inf')):
    """
    Corrects the tlwh bounding box format if top-left coordinates are negative.

    Parameters:
        tlwh (tuple): A tuple (top-left x, top-left y, width, height) of the bounding box.
        image_width (int): The width of the image.
        image_height (int): The height of the image.

    Returns:
        tuple: A corrected bounding box (top-left x, top-left y, width, height).
    """
    x, y, w, h = tlwh

    # Ensure the top-left corner is within the image boundaries
    new_x = int(max(0, x))
    new_y = int(max(0, y))

    # Adjust width and height if the original x or y were negative
    if x < 0:
        w += x  # Reduce width by the amount x was out of bounds
    if y < 0:
        h += y  # Reduce height by the amount y was out of bounds

    # Ensure width and height do not exceed image boundaries
    w = int(min(w, image_width - new_x))
    h = int(min(h, image_height - new_y))

    return (new_x, new_y, w, h)

def parse_mot_results(filename, skip_frame_num=5):
    basename = os.path.basename(filename)
    
    results = []
    with open(filename, 'r') as file:
        # Read the file line by line
        for line in file:
            # Split each line by comma or space (depends on your file format)
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue  # Ensure there are enough columns in this line

            # Convert each part to the appropriate type
            frame_id = int(parts[0])
            if frame_id % skip_frame_num != 0:
                continue
            
            track_id = int(parts[1])
            bbox_x = float(parts[2])
            bbox_y = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            confidence = float(parts[6]) if len(parts) > 6 else None
            class_id = int(parts[7]) if len(parts) > 7 else None
            visibility = float(parts[8]) if len(parts) > 8 else None

            # Create a dictionary for the current tracking result
            track_result = {
                "camera": basename,
                'frame_id': frame_id,
                'track_id': track_id,
                'tlwh': correct_tlwh((bbox_x, bbox_y, width, height)),
                'confidence': confidence,
            }
            if class_id is not None:
                track_id[class_id] = class_id
            if visibility is not None:
                track_id[visibility] = visibility

            # Add the dictionary to the results list
            results.append(track_result)

    return results

""" Paddle """
def load_mot_result(video_name = "IMG_2230", crop_folder=None):
    video_path = f"../data/mot/{video_name}.mp4"
    mot_res_path = f"../data/mot/{video_name}.csv"

    df = pd.read_csv(mot_res_path)
    df = df[df.frame_id % 10 == 0]
    df.tlwh = df.tlwh.map(eval)
    df.feature = df.feature.apply(eval)
    df.loc[:, 'video'] = video_name

    # get crops
    if crop_folder:
        cropped_images = crop_targets_from_video(
            video_path, 
            df.to_dict(orient='records'), 
            output_dir = crop_folder + f'/{video_name}')
        # df['crops'] = cropped_images

    return df


""" Aux func """
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

    # Process each crop specification
    for crop in crops:
        frame_id = crop['frame_id']
        x1, y1, w, h = tlwh = crop['tlwh']
        
        # Check if the frame_id is valid
        if frame_id >= frame_count:
            continue

        # Set the video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        # Crop the image
        crop_img = frame[max(0, y1): y1 + h, max(x1, 0): x1 + w]
        # crop_img = frame
        # crop_img = cv2.resize(crop_img, dsize=None, fx=.57, fy=.57)
        # cv2.rectangle(crop_img, (x1, y1), (x1 + w, y1 + h), color=(0, 255, 0), thickness=2)

        if output_dir:
            img_fn = os.path.join(output_dir, f"{crop['track_id']:03d}_{frame_id:04d}.jpg")
            try:
                cv2.imwrite(img_fn, crop_img)
            except:
                logger.error(img_fn)
        
        cropped_images.append(crop_img)

    cap.release()
    return cropped_images


# %%

# 获取 crops 
videos = ['IMG_2230', 'IMG_2231', 'IMG_2232']
res = []
for video in videos:
    df = load_mot_result(video_name = video) # , crop_folder="../data/crops"
    res.append(df)
    
df_all = pd.concat(res)

#%%
def append_crop_fn(item, folder):
    fn = f"{item['video']}/{item['track_id']:03d}_{item['frame_id']:04d}.jpg"
    return os.path.join(folder, fn)

df_all['crop_fn'] = df_all.apply(lambda x: append_crop_fn(x, "../data/crops"), axis=1)
df_all


#%%
from faceDetect import FaceDetector

recognizer = FaceDetector(gallery_path='../data/gallery')

#%%
res = []
for fn in df_all.crop_fn:
    crop = cv2.imread(fn)
    res.append(recognizer.detect_and_identify(crop))
len(res)

#%%
video_name = videos[2]
df = df_all.query("video == @video_name")

#%%
import seaborn as sns
from metric.similarity import cosine_similarity, get_dist_mat

feats = np.vstack(df.feature)
norms = np.linalg.norm(feats, axis=1, keepdims=True)
feats = feats / norms

dis_mat = np.abs(1 - get_dist_mat(feats, feats, func_name='cosine'))
sns.heatmap(dis_mat)


# %%
from sklearn.cluster import AgglomerativeClustering
cluster1 = AgglomerativeClustering(
    n_clusters=None,
    distance_threshold=0.5,
    metric='precomputed',
    linkage='complete')
cluster_labels1 = cluster1.fit_predict(dis_mat)

cluster_labels1

# %%
import hdbscan
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def get_match(cluster_labels):
    cluster_dict = dict()
    cluster = list()
    for i, l in enumerate(cluster_labels):
        if l in list(cluster_dict.keys()):
            cluster_dict[l].append(i)
        else:
            cluster_dict[l] = [i]
    for idx in cluster_dict:
        cluster.append(cluster_dict[idx])
    return cluster

def cluster_by_hdbscan(M):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True, metric='precomputed')
    clusterer.fit(M)

    labels = clusterer.labels_

    return labels

def cluster_by_dbscan(M):
    clusterer = DBSCAN(eps=0.18, min_samples=1, metric='precomputed')
    clusterer.fit(M)

    labels = clusterer.labels_

    return labels

cluster_by_dbscan(dis_mat)

# %%
cluster_by_hdbscan(dis_mat)



# %%
