#%%
import os
import cv2
import insightface
import numpy as np
import matplotlib.pyplot as plt

from .utils.logger_helper import logger


class FaceDetector:
    def __init__(self, model_type='buffalo_l', gallery_path=None, device='cpu'):
        # 加载人脸识别模型
        self.model = insightface.app.FaceAnalysis(model_type)
        self.model.prepare(ctx_id=self.set_device(device))
        
        self.gallery = {}
        if gallery_path:
            self.load_gallery(gallery_path)

    def set_device(self, device):
        if device == 'cpu':
            return -1
        elif device.startswith('cuda'):  # 设备为cuda时
            gpu_index = int(device.split(':')[-1]) if ':' in device else 0
            return gpu_index
        return 0
            
    def load_gallery(self, gallery_path):
        """从指定文件夹加载人脸数据到画廊"""
        try:
            for filename in os.listdir(gallery_path):
                if not filename.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                
                path = os.path.join(gallery_path, filename)
                username = filename.split('.')[0]
                face = self.detect(cv2.imread(path))
                if face is None:
                    continue
                
                if face['embedding'] is not None:
                    self.gallery[username] = face['embedding']
                else:
                    print(f"Warning: No face detected in {filename}.")

        except Exception as e:
            logger.error(f"Failed to load gallery from {gallery_path}: {str(e)}")        

    def get_embedding(self, image):
        """提取单个人脸的特征向量"""
        faces = self.model.get(image)
        if faces:
            return faces[0].embedding
        return None

    def detect(self, image, norm=True):
        """检测单个图像中的最优人脸"""
        faces = self.model.get(image)
        
        if faces:
            # 选择置信度最高或者bbox面积最大的人脸
            best_face = max(faces, key=lambda x: x.det_score)  
            x1, y1, x2, y2 = best_face.bbox.astype(int)
            tlwh = (x1, y1, x2 - x1, y2 - y1)
            if norm is False:
                embedding = best_face.embedding
            else :
                embedding = best_face.embedding / np.linalg.norm(best_face.embedding)
            return {
                'bbox': (x1, y1, x2, y2),
                'tlwh': tlwh,
                'embedding': embedding,
                'confidence': best_face.det_score
            }
        
        return None

    def match_face(self, emb1, emb2):
        """计算两个人脸特征向量之间的相似度"""
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        similarity = dot_product / (norm_emb1 * norm_emb2)
        return similarity

    def show_face(self, image, result):
        """在图像上绘制人脸框并显示"""
        if result is not None:
            x1, y1, x2, y2 = result['bbox']
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            print("No face detected in the image.")

    def identify_face(self, image, match_threshold=0.7):
        """识别图像中的人脸是否与画廊中的某个人匹配"""
        result = self.detect(image)
        if result:
            best_match = None
            highest_similarity = 0
            face_encoding = result['embedding']

            for username, emb in self.gallery.items():
                similarity = self.match_face(face_encoding, emb)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = username

            if highest_similarity > match_threshold:
                logger.info(f"best macth: {best_match}, sim: {highest_similarity*100:.1f}%")
                return {
                    'username': best_match,
                    'similarity': highest_similarity
                }

        logger.debug(f"Unmacth, the highest simimarity: {similarity*100:.1f}%")

        return None


if __name__ == "__main__":
    recognizer = FaceDetector(gallery_path='../data/gallery')

    image = cv2.imread('../data/obama-test1.jpeg')
    result = recognizer.detect(image)
    recognizer.show_face(image, result)
    recognizer.identify_face(image)

    # trump    
    image = cv2.imread('../data/trump.jpeg')
    recognizer.identify_face(image)
    result = recognizer.detect(image)
    recognizer.show_face(image, result)


# %%
