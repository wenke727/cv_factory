#%%
from pathlib import Path
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
    user_gallery['appearance_filenames'] = appearance_filenames
    user_gallery['appearance_gallery'] = appearance_gallery

    return user_gallery


if __name__ == "__main__":
    # 加载人脸识别模型
    face_detector = FaceDetector()
    reid_model = load_ReID_model(model_dir='../ckpt/reid_model')
    folder = Path("../data/gallery/wenke")

    user_gallery = load_user_gallery(face_detector, reid_model, folder)
    save_checkpoint(user_gallery, folder / "gallery.ckpt")

    user_gallery


# %%
