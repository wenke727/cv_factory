# test_face_recognizer.py

import pytest
import cv2
from unittest.mock import Mock
from src.faceDetect import FaceDetector

@pytest.fixture
def sample_image_path():
    return './data/obama-test1.jpeg'  # 指向包含测试面部的图像文件

@pytest.fixture
def recognizer():
    # 创建 FaceRecognizer 实例，使用测试用的画廊路径
    return FaceDetector(gallery_path='./data/gallery', device='cpu')

def test_load_gallery(recognizer):
    # 测试画廊是否加载正确
    assert len(recognizer.gallery) > 0, "Gallery should not be empty"
    assert 'obama' in recognizer.gallery, "Obama should be in the gallery"

def test_detect_face(recognizer, sample_image_path):
    # 测试是否能检测到人脸
    image = cv2.imread(sample_image_path)
    result = recognizer.detect(image)
    assert result is not None, "Should detect a face"
    assert result['confidence'] > 0.5, "Face detection confidence should be high"

def test_identify_face(recognizer, sample_image_path):
    # 测试人脸识别功能
    image = cv2.imread(sample_image_path)
    identification = recognizer.identify_face(image)
    assert identification is not None, "Should identify a face"
    assert identification['username'] == 'obama', "Should identify Obama"

# def test_show_face(recognizer, sample_image_path, mocker):
#     # 测试显示人脸功能
#     mocker.patch('matplotlib.pyplot.show')  # 模拟plt.show()以避免实际显示图像
#     image = cv2.imread(sample_image_path)
#     result = recognizer.detect(image)
#     recognizer.show_face(image, result)
#     # 这个测试确保没有异常抛出，并且绘图函数被调用
#     assert True, "Should execute show_face without errors"
