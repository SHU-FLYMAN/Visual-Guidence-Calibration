import os
import cv2
import numpy as np
from utils import imwrite2, imread2
from tqdm import tqdm
from GrayCode import GrayCode
from scipy.io import savemat, loadmat


# 相移法实现
class Phaser(object):
    def __init__(self):
        pass

    def load_config(self, N, n, BT, iter_num, gamma_file=None):
        self.iter_num = iter_num
        self.N = N
        self.n = n
        self.BT = BT
        self._grayCode = GrayCode()
        # 加载gamma文件
        self.gamma_file = gamma_file
        if self.gamma_file is not None:
            g = loadmat(gamma_file)
            self.a, self.b, self.c = g["a"], g["b"], g["c"]

    def makePatterns(self, A, B, N, T, W, H, save_dir, flip=False):
        xs = np.arange(W)
        f = W / T
        for k in range(N):
            I = A + B * np.cos((2. * f * xs / W - 2. * k / N) * np.pi)
            img = np.tile(I, (H, 1))
            if flip:
                img = img.T
            # 不返回，因为太耗费内存
            imwrite2(save_dir, img)

    def _calcWrappedPhase(self, files):
        N = len(files)
        sin_sum, cos_sum = 0., 0.
        for k in range(N):
            Ik = self._loadIk(files[k])
            pk = 2. * k / N * np.pi
            sin_sum += Ik * np.sin(pk)
            cos_sum += Ik * np.cos(pk)
        pha = np.arctan2(sin_sum, cos_sum)
        B = np.sqrt(sin_sum ** 2 + cos_sum ** 2) * 2 / N
        e = -0.00000000001
        pha_low_mask = pha < e
        pha = pha + pha_low_mask * 2. * np.pi
        return pha, B

    def gauss_iter(self, pha, N, iter_num):
        A = 0.5
        B = 0.5
        bias = 0.0
        ker_size = 7
        pha_c = np.copy(pha)
        for i in tqdm(range(iter_num)):
            phas = []
            for k in range(N):
                p = A + B * np.cos(pha_c + bias + 2 * k * np.pi / N)
                p = cv2.GaussianBlur(p, (ker_size, ker_size), 0)
                phas.append(p)
            sin_sum = 0.
            cos_sum = 0.
            for k in range(N):
                sin_sum += phas[k] * np.sin(2 * k * np.pi / N)
                cos_sum += phas[k] * np.cos(2 * k * np.pi / N)
            pha_c = np.arctan2(sin_sum, cos_sum)
        pha_c[np.isnan(pha_c)] = 0.
        return pha_c

    def gauss_iter2(self, pha, iter_num):
        pha_c = np.copy(pha)
        ker_size = 7
        for i in range(iter_num):
            pha_c = cv2.GaussianBlur(pha_c, (ker_size, ker_size), 0)
        pha_c[np.isnan(pha_c)] = 0.
        return pha_c

    def _loadIk(self, file):
        Ik = imread2(file).astype(np.float64) / 255.
        if self.gamma_file is not None:
            Ik = self.gamma_correct(Ik, self.a, self.b, self.c)
        return Ik

    # 输入输出必须都在255范围内
    def gamma_correct(self, img, a, b, c):
        max_v = np.max(img)
        if max_v > 1:
            raise ValueError("先归一化灰度值")
        return np.power((img - c) / a, 1 / b)

    def calcAbsolutePhase(self, files):
        files_phase, files_gray = files[: self.N], files[self.N:]
        # 01 相移法
        pha_wrapped, B = self._calcWrappedPhase(files_phase)
        # 02 互补格雷码解条纹阶次
        KS1, KS2 = self._grayCode.calcGrayCode(files_gray, self.n)
        # 03 计算最终相位
        mask1 = pha_wrapped <= np.pi / 2
        mask2 = np.multiply(np.pi / 2 < pha_wrapped, pha_wrapped <= 3 / 2 * np.pi)
        mask3 = pha_wrapped > 3 / 2 * np.pi
        pha = np.multiply(pha_wrapped + 2 * np.pi * KS2, mask1) \
            + np.multiply(pha_wrapped + 2 * np.pi * KS1, mask2) \
            + np.multiply(pha_wrapped + 2 * KS2 * np.pi - 2 * np.pi, mask3)
        # 调制度过滤
        pha = np.multiply(pha, B >= self.BT)
        # 归一化
        pha_absolute = pha  / (2 * np.pi * (2 ** self.n))

        pha_absolute_iter = 0
        #pha_absolute_iter2 = 0
        # 04 高斯迭代滤波
        if self.iter_num > 0:
            # pha_absolute_iter = self.gauss_iter(pha_absolute, self.N, self.iter_num)
            pha_absolute_iter = self.gauss_iter2(pha_absolute, self.iter_num)

        d = {"pha_absolute": pha_absolute,
             "pha_absolute_iter": pha_absolute_iter,
             #"pha_absolute_iter2": pha_absolute_iter2,
             "pha_wrapped": pha_wrapped,
             "KS1": KS1,
             "KS2": KS2,
             "B": B}
        return d
