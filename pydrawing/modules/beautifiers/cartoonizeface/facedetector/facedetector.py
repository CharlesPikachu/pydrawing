'''
Function:
    人脸检测
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import cv2
import ssl
import math
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context


'''FaceDetector'''
class FaceDetector():
    def __init__(self, use_cuda, **kwargs):
        super(FaceDetector, self).__init__()
        try:
            import face_alignment
        except:
            raise RuntimeError('Please run "pip install face_alignment" to install "face_alignment"')
        device = 'cuda' if use_cuda else 'cpu'
        self.dlib_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device=device, face_detector='dlib')
    '''forward'''
    def __call__(self, image):
        # obtain landmarks
        preds = self.dlib_detector.get_landmarks(image)
        landmarks = None
        if preds is None:
            raise RuntimeError('no faces are detected')
        elif len(preds) == 1:
            landmarks = preds[0]
        else:
            areas = []
            for pred in preds:
                landmarks_top = np.min(pred[:, 1])
                landmarks_bottom = np.max(pred[:, 1])
                landmarks_left = np.min(pred[:, 0])
                landmarks_right = np.max(pred[:, 0])
                areas.append((landmarks_bottom - landmarks_top) * (landmarks_right - landmarks_left))
            max_face_index = np.argmax(areas)
            landmarks = preds[max_face_index]
        # rotate
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        radian = np.arctan((left_eye_corner[1] - right_eye_corner[1]) / (left_eye_corner[0] - right_eye_corner[0]))
        height, width, _ = image.shape
        cos = math.cos(radian)
        sin = math.sin(radian)
        new_w = int(width * abs(cos) + height * abs(sin))
        new_h = int(width * abs(sin) + height * abs(cos))
        Tx = new_w // 2 - width // 2
        Ty = new_h // 2 - height // 2
        M = np.array([[cos, sin, (1 - cos) * width / 2. - sin * height / 2. + Tx], [-sin, cos, sin * width / 2. + (1 - cos) * height / 2. + Ty]])
        image_rotate = cv2.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))
        landmarks = np.concatenate([landmarks, np.ones((landmarks.shape[0], 1))], axis=1)
        landmarks_rotate = np.dot(M, landmarks.T).T
        # return
        return self.crop(image_rotate, landmarks_rotate)
    '''crop'''
    def crop(self, image, landmarks):
        landmarks_top = np.min(landmarks[:, 1])
        landmarks_bottom = np.max(landmarks[:, 1])
        landmarks_left = np.min(landmarks[:, 0])
        landmarks_right = np.max(landmarks[:, 0])
        top = int(landmarks_top - 0.8 * (landmarks_bottom - landmarks_top))
        bottom = int(landmarks_bottom + 0.3 * (landmarks_bottom - landmarks_top))
        left = int(landmarks_left - 0.3 * (landmarks_right - landmarks_left))
        right = int(landmarks_right + 0.3 * (landmarks_right - landmarks_left))
        if bottom - top > right - left:
            left -= ((bottom - top) - (right - left)) // 2
            right = left + (bottom - top)
        else:
            top -= ((right - left) - (bottom - top)) // 2
            bottom = top + (right - left)
        image_crop = np.ones((bottom - top + 1, right - left + 1, 3), np.uint8) * 255
        h, w = image.shape[:2]
        left_white = max(0, -left)
        left = max(0, left)
        right = min(right, w-1)
        right_white = left_white + (right-left)
        top_white = max(0, -top)
        top = max(0, top)
        bottom = min(bottom, h-1)
        bottom_white = top_white + (bottom - top)
        image_crop[top_white:bottom_white+1, left_white:right_white+1] = image[top:bottom+1, left:right+1].copy()
        return image_crop