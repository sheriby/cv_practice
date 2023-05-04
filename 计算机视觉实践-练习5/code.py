from PIL import Image
from pylab import *
import cv2
import imageio
from numpy import *
from numpy.ma import array
from scipy.ndimage import uniform_filter

def plane_sweep_ncc(im_l,im_r,start,steps,wid):
    m,n = im_l.shape
    # 保存不同求和值的数组
    mean_l = zeros((m,n))
    mean_r = zeros((m,n))
    s = zeros((m,n))
    s_l = zeros((m,n))
    s_r = zeros((m,n))
    # 保存深度平面的数组
    dmaps = zeros((m,n,steps))
    # 计算图像块的平均值
    uniform_filter(im_l,wid,mean_l)
    uniform_filter(im_r,wid,mean_r)
    # 归一化图像
    norm_l = im_l - mean_l
    norm_r = im_r - mean_r
    # 尝试不同的视差
    for displ in range(steps):
        # 将左边图像移动到右边，计算加和
        uniform_filter(np.roll(norm_l, -displ - start) * norm_r, wid, s) # 和归一化
        uniform_filter(np.roll(norm_l, -displ - start) * np.roll(norm_l, -displ - start), wid, s_l)
        uniform_filter(norm_r*norm_r,wid,s_r) # 和反归一化
        # 保存 ncc 的分数
        dmaps[:,:,displ] = s / sqrt(s_l * s_r)
        # 为每个像素选取最佳深度
    return np.argmax(dmaps, axis=2)

im_l = array(Image.open('left.png').convert('L'), 'f')
im_r = array(Image.open('right.png').convert('L'),'f')
# 开始偏移，并设置步长
steps = 40
start = 4
# ncc 的宽度
wid = 16

res = plane_sweep_ncc(im_l,im_r,start,steps,wid)
res = res * (255 / np.max(res)) 
imageio.imsave('result_16.png', np.uint8(res))

