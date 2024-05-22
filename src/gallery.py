#%%
import glob
import numpy as np
from pathlib import Path
from loguru import logger

from algs.faceDetect import FaceDetector
from algs.ReID import encode_folder_crops_to_reid, load_ReID_model

from utils_helper.serialization import save_checkpoint


def load_user_gallery(face_detector, reid_model, user_folder):
    """
    Loads a user gallery from given folder paths for face and appearance data.

    Parameters:
        face_detector: An instance of the FaceDetector class.
        reid_model: An instance of a ReID model loaded.
        user_folder: Path object or str, the folder containing the user's data.

    Returns:
        dict: A dictionary containing face and appearance gallery data.
    """
    user_folder = Path(user_folder)

    user_gallery = {}
    face_folder = user_folder / 'face'
    face_detector.reset_gallery()
    face_gallery = face_detector.load_gallery(face_folder)

    # Ensure there is exactly one face in the gallery
    if len(face_gallery) != 1:
        raise ValueError("Only one face per user is supported")

    # Store face gallery information
    face_filename, face_data = next(iter(face_gallery.items()))
    user_gallery['face_filename'] = face_filename
    user_gallery['face_gallery'] = face_data

    # Load appearance gallery
    appearance_folder = user_folder / 'appearance'
    appearance_filenames, appearance_gallery = encode_folder_crops_to_reid(
        reid_model, appearance_folder, is_normed=True)

    # Store appearance gallery information
    logger.debug(f"appearance lengh: {len(appearance_filenames)}")
    user_gallery['appearance_filenames'] = appearance_filenames
    user_gallery['appearance_gallery'] = appearance_gallery

    return user_gallery


def load_user_galleries(face_detector, person_reid_model, base_folder):
    """
    Load and concatenate galleries for multiple users.

    Parameters:
        face_detector: An instance of FaceDetector class.
        reid_model: An instance of a ReID model.
        base_folder: str, path to the folder containing all user folders.

    Returns:
        tuple: Contains concatenated face and appearance galleries and corresponding user mappings.
    """
    # Generate list of user names and load galleries
    base_path = Path(base_folder)
    user_folders = [folder for folder in base_path.iterdir() if folder.is_dir()]  # 过滤仅包含目录
    user_names = [folder.name for folder in user_folders]
    user_galleries = [load_user_gallery(face_detector, person_reid_model, str(folder))
                        for folder in user_folders]

    # Concatenate galleries
    appearance_gallery = np.concatenate([user['appearance_gallery'] for user in user_galleries])
    face_gallery = np.concatenate([user['face_gallery'][np.newaxis, :] for user in user_galleries])

    # Create index to user mappings
    appearance_index_to_user = np.concatenate([[user] * len(user_gallery['appearance_gallery'])
                                            for user, user_gallery in zip(user_names, user_galleries)])
    face_index_to_user = np.concatenate([[user] for user, _ in zip(user_names, user_galleries)])

    return face_gallery, face_index_to_user, appearance_gallery, appearance_index_to_user


if __name__ == "__main__":
    # 加载人脸识别模型
    face_detector = FaceDetector()
    person_reid_model = load_ReID_model(model_dir='../ckpt/reid_model')

    face_gallery, face_index_to_user, appearance_gallery, appearance_index_to_user = \
        load_user_galleries(face_detector, person_reid_model, base_folder="../data/gallery")
    gallery = {
        "face_gallery": face_gallery,
        "face_index_to_user": face_index_to_user,
        "appearance_gallery": appearance_gallery,
        "appearance_index_to_user": appearance_index_to_user
    }
    save_checkpoint(gallery, '../data/gallery/gallery.ckpt')


# %%
