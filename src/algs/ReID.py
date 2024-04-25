import sys
import cv2
import glob
import numpy as np

import sys
sys.path.append('/Users/wenke/Library/CloudStorage/OneDrive-Personal/3_Codes/PaddleDetection/deploy')

from pipeline.pphuman.reid import ReID


def load_ReID_model(model_dir='/root/pcl/PedFactory/weights/reid_model', device="CPU"):
    model = ReID(model_dir=model_dir, device=device, batch_size=16)

    return model

def encode_folder_crops_to_reid(model, folder, is_normed=False):
    fns = sorted(glob.glob(f'{folder}/*.jpg'))
    feats = encode_crops_to_reid(model, fns, is_normed)

    return fns, feats

def encode_crops_to_reid(model, fns, is_normed=False):
    imgs = [cv2.imread(i) for i in fns]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

    feats = model.predict_batch(imgs)
    feats = np.array(feats)

    if not is_normed:
        feats = feats / np.linalg.norm(feats, 2, axis=1, keepdims=True)

    return feats


if __name__ == "__main__":
    model  = load_ReID_model()
    model = ReID('../../ckpt/reid_model', 'cpu')

# %%
