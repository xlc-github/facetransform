import cv2,dlib
import matplotlib.pyplot as plt
import numpy as np
#先做一个人脸裁剪，然后再进行人脸对齐，主要是根据人眼位置进行对齐，做一个仿射变换，达到对齐的效果
predictor_model = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()# dlib人脸检测器
predictor = dlib.shape_predictor(predictor_model)

# cv2读取图像
test_img_path = "../img/3.jpg"
img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 人脸数rects
rects = detector(img, 0)
# faces存储full_object_detection对象
faces = dlib.full_object_detections()
def face_crop():
    for i in range(len(rects)):
        faces.append(predictor(img,rects[i]))
        # print(rects[i])

    face_images = dlib.get_face_chips(img, faces, size=320)
    for image in face_images:
        cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imwrite('../output/Messi_clip.png', cv_bgr_img)
    return cv_bgr_img


import dlib
import face_recognition
import math
import numpy as np
import cv2
import sys
import os
from os.path import basename







def face_alignment(face_img):
    # 预测关键点
    # print("进行对齐-----")
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    rec = dlib.rectangle(0, 0, face_img.shape[0], face_img.shape[1])
    shape = predictor(np.uint8(face_img), rec)
    # left eye, right eye, nose, left mouth, right mouth
    order = [36, 45, 30, 48, 54]
    # for j in order:
    #     x = shape.part(j).x
    #     y = shape.part(j).y
    # 计算两眼的中心坐标
    eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
    dx = (shape.part(45).x - shape.part(36).x)
    dy = (shape.part(45).y - shape.part(36).y)
    # 计算角度
    angle = math.atan2(dy, dx) * 180. / math.pi
    # 计算仿射矩阵
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    # 进行仿射变换，即旋转
    RotImg = cv2.warpAffine(face_img, RotateMatrix, (face_img.shape[0], face_img.shape[1]))

    return RotImg

if __name__ == '__main__':
    img = cv2.imread("../img/face_clip.png")
    align_img = face_alignment(img)
    cv2.imwrite("../output/align.jpg",align_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

