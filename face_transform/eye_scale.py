
import cv2
import dlib
import math
import numpy as np

predictor_path = 'shape_predictor_68_face_landmarks.dat'
# 使用dlib自带的frontal_face_detector作为我们的特征提取器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#得到68个人脸特征点
def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        # for idx,point in enumerate(land_marks_node):
        #     # 68点坐标
        #     pos = (point[0,0],point[0,1])
        #     print(idx,pos)
        #     # 利用cv2.circle给每个特征点画一个圈，共68个
        #     cv2.circle(img_src, pos, 5, color=(0, 255, 0))
        #     # 利用cv2.putText输出1-68
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(img_src, str(idx + 1), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        land_marks.append(land_marks_node)

    return land_marks


'''
方法： Interactive Image Warping 局部平移算法
'''



def eye_scale_auto(srcImg,radius,cen_x,cen_y,intensity):

    copyImg = srcImg.copy()
    H, W, C = srcImg.shape
    ddradius = radius*radius
    k0 = intensity/100.0
    for i in range(W):
        for j in range(H):
            #先判断改点是否在眼睛所处的圆形区域内
            if (i-cen_x)*(i-cen_x)+(j-cen_y)*(j-cen_y) > ddradius:
                continue
            dis = (i-cen_x)*(i-cen_x)+(j-cen_y)*(j-cen_y)#圆内一点到圆心的距离平方
            if dis < ddradius:#遍历的每个点都在处理区域以内，圆内
                # 计算出（i,j）坐标的原坐标
                k = 1.0 - (1.0-dis/ddradius)*k0
                xd = (i - cen_x)*k+cen_x
                yd = (j - cen_y)*k+cen_y
                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, xd, yd)
                # 改变当前 i ，j的值
                copyImg[j, i] = value
    return copyImg

# 双线性插值法
def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


def eye_auto(src,intensity):
    landmarks = landmark_dec_dlib_fun(src)

    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        #左眼
        left_eye_start = landmarks_node[36]
        left_eye_end = landmarks_node[39]

        #右眼
        right_eye_start = landmarks_node[42]
        right_eye_end = landmarks_node[45]

        D_l = math.sqrt(
            (left_eye_start[0,0]-left_eye_end[0,0])*(left_eye_start[0,0]-left_eye_end[0,0])+
            (left_eye_start[0, 1] - left_eye_end[0, 1]) * (left_eye_start[0, 1] - left_eye_end[0, 1])
        )

        D_r = math.sqrt(
            (right_eye_start[0, 0] - right_eye_end[0, 0]) * (right_eye_start[0, 0] - right_eye_end[0, 0]) +
            (right_eye_start[0, 1] - right_eye_end[0, 1]) * (right_eye_start[0, 1] - right_eye_end[0, 1])
        )
        Radius = max(D_l,D_r)

        #计算两眼中心位置

        centX_l,centY_l =(left_eye_start[0,0]+left_eye_end[0,0])/2,(left_eye_start[0, 1]+left_eye_end[0, 1])/2

        centX_r, centY_r = (right_eye_start[0, 0] + right_eye_end[0, 0]) / 2, (right_eye_start[0, 1] + right_eye_end[0, 1]) / 2

        # 左眼
        left_eye_scale = eye_scale_auto(src, Radius,centX_l,centY_l,intensity)
        # 右眼
        right_eye_scale = eye_scale_auto(left_eye_scale, Radius,centX_r,centY_r,intensity)

    # 显示
    cv2.imshow('eyescale', right_eye_scale)
    cv2.imwrite('thin0.jpg', right_eye_scale)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    src = cv2.imread('../img/3.jpg')
    cv2.imshow('src', src)
    eye_auto(src,10)
    cv2.waitKey(50)

#
if __name__ == '__main__':
    main()
