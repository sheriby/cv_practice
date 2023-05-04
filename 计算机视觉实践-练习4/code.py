import cv2
import numpy as np
import pylab as plt

if __name__ == '__main__':
    # 读取原始图像
    im_src = cv2.imread('img.jpg')
    h, w, c = im_src.shape

    # 原始图像中物体的四个顶点的信息
    pts_src = np.array([(0, 0), (1280, 0), (0, 800), (1280, 800)])
    # 目标物体中的物体的四个顶点信息
    pts_dst = np.array([(265, 30), (796, 99), (100, 473), (932, 373)])

    # 计算单应性矩阵 Homography
    # 是一个3x3的矩阵，根据对应的两个点，计算出变换矩阵，由此将原始图像进行转换。
    homography, status = cv2.findHomography(pts_src, pts_dst)
    print(homography.shape)
    print(homography)

    # 基于单应性矩阵，将原始图像转换成目标图像
    im_out = cv2.warpPerspective(im_src, homography, (w, h))

    plt.figure()
    plt.subplot(1, 2, 1), plt.imshow(im_src[:, :, ::-1]), plt.title('src')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(im_out[:, :, ::-1]), plt.title('out')
    plt.xticks([]), plt.yticks([])

    plt.show()  # show dst