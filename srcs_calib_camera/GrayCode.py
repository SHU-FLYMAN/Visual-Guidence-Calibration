import numpy as np
from utils import imwrite2, imread2, expand_dim
from tqdm import tqdm


def calcKS(gcs, n, v, u, V1K1, V2K2):
    gc1 = gcs[:n, v, u]
    gc2 = gcs[:n + 1, v, u]
    v1 = int(np.sum([gc1[i] * 2 ** (n - (i + 1)) for i in range(n)]))
    v2 = int(np.sum([gc2[i] * 2 ** (n + 1 - (i + 1)) for i in range(n + 1)]))
    return V1K1[v1], V2K2[v2]


class GrayCode(object):
    def makePatterns(self, n, W, H, save_dir, flip=False):
        patterns = []
        # 01 格雷码
        codes = self._GrayCode(n)
        num = 0  # 格雷码的数量
        for idx in range(n):
            row = codes[idx, :]
            one_row = np.zeros([W], np.uint8)
            num = len(row)
            assert (W % num == 0)  # 必须整除
            per_col = int(W / num)

            for i in range(num):
                one_row[i * per_col: (i + 1) * per_col] = row[i]
            pattern = np.tile(one_row, (H, 1)) * 255
            patterns.append(pattern)

        # 02 互补格雷码
        codes_com = self._GrayCodeCom(n)
        num2 = len(codes_com)  # 互补格雷码的数量
        assert (W % num2 == 0)
        per_col = int(W / num2)
        one_row = np.zeros([W], np.uint8)
        for i in range(num2):
            one_row[i * per_col: (i + 1) * per_col] = codes_com[i]
        pattern = np.tile(one_row, (H, 1)) * 255
        patterns.append(pattern)

        img_white = np.ones_like(pattern, dtype=np.uint8) * 255
        img_black = np.zeros_like(pattern, dtype=np.uint8)

        patterns.append(img_white)
        patterns.append(img_black)
        # 03 查看结果
        width = 20
        img = np.zeros(((n + 1) * width, num2 * width), dtype=np.uint8)
        for idx in range(codes.shape[0]):
            one_row = np.zeros([num2 * width], np.uint8)
            for i in range(num):
                one_row[i * width * 2: (i + 1) * width * 2] = codes[idx, i]
            img[idx * width: (idx + 1) * width, :] = np.tile(one_row, (width, 1)) * 255
        one_row = np.zeros([num2 * width], np.uint8)
        for i in range(num2):
            one_row[i * width: (i + 1) * width] = codes_com[i]
        img[n * width: (n + 1) * width, :] = np.tile(one_row, (width, 1)) * 255

        # 04 写入图像
        for pattern in patterns:
            if flip:
                pattern = pattern.T
            imwrite2(save_dir, pattern)

    def calcGrayCode(self, files, n):
        # 01 读取图片
        num = len(files)
        img = imread2(files[0])
        h, w = img.shape
        Is = np.zeros((num, h, w), dtype=np.float32)
        for idx, file in enumerate(files):
            Is[idx, ...] = imread2(file).astype(np.float32) / 255.

        # 02 计算Is_Max、Is_Min，对每个点进行阈值判断,计算出编码值
        Is_max = np.max(Is, axis=0)
        Is_min = np.min(Is, axis=0)
        Is_std = (Is - Is_min) / (Is_max - Is_min)
        gcs = Is_std > 0.5
        gcs = gcs.astype(np.int32)

        # 04 开始解码KS1 KS2
        K1S, K2S = self._calcKS(gcs, n, h, w)
        return K1S, K2S


    def _calcKS(self, gcs, n, h, w):
        V1K1 = self._V1toK1(n)
        V2K2 = self._V2toK2(n)

        gcs1 = gcs[:n, ...]
        gcs2 = gcs[:n + 1, ...]
        ns1 = expand_dim(np.arange(n))
        ns2 = expand_dim(np.arange(n + 1))
        VS1 = np.sum(gcs1 * np.power(2, (n - (ns1 + 1))), axis=0)
        VS2 = np.sum(gcs2 * np.power(2, (n + 1 - (ns2 + 1))), axis=0)

        # 映射到KS
        KS1 = np.zeros((h, w), dtype=np.int32)
        KS2 = np.zeros((h, w), dtype=np.int32)
        for v in tqdm(range(h), "解码格雷码"):
            for u in range(w):
                KS1[v, u] = V1K1[VS1[v, u]]
                KS2[v, u] = V2K2[VS2[v, u]]
        return KS1, KS2

    def _V1toK1(self, n):
        graycode = self._GrayCode(n)
        # 03 建立V1 -> K1 的映射表
        V1_ROW = []
        for i in range(2 ** n):
            code = graycode[:, i]
            v1 = 0
            for j in range(n):
                v1 += code[j] * 2 ** (n - (j + 1))
            V1_ROW.append(v1)
        V1K = dict()
        for idx, v1 in enumerate(V1_ROW):
            V1K[v1] = idx
        return V1K

    def _V2toK2(self, n):
        graycode = self._GrayCode(n)
        # 04 建立V2 -> K2 的映射表
        V2_ROW = []
        graycode2 = np.repeat(graycode, 2, axis=1)
        graycode_com = self._GrayCodeCom(n)
        graycode_all = np.vstack((graycode2, graycode_com))

        for i in range(2 ** (n + 1)):
            code = graycode_all[:, i]
            v2 = 0
            for j in range(n + 1):
                v2 += code[j] * 2 ** (n + 1 - (j + 1))
            V2_ROW.append(v2)

        V2K = dict()
        for idx, v2 in enumerate(V2_ROW):
            V2K[v2] = int((idx + 1) / 2)
        return V2K

    def _GrayCode(self, n:int):
        code_temp = GrayCode.__GrayCode(n)
        codes = []
        for row in range(len(code_temp[0])):
            c = []
            for idx in range(len(code_temp)):
                c.append(int(code_temp[idx][row]))
            codes.append(c)
        return np.array(codes, np.uint8)

    @staticmethod
    def __GrayCode(n:int):
        if n < 1:
            print("输入数字必须大于0")
            assert (0);
        elif n == 1:
            code = ["0", "1"]
            return code
        else:
            code = []
            code_pre = GrayCode.__GrayCode(n - 1)
            for idx in range(len(code_pre)):
                code.append("0" + code_pre[idx])
            for idx in range(len(code_pre) - 1, -1, -1):
                code.append("1" + code_pre[idx])
            return code

    def _GrayCodeCom(self, n:int):
        gcs5 = [0, 1, 1, 0]
        num = 2 ** (n + 1)
        codes_com = []
        for i in range(num):
            r = i % 4
            codes_com.append(gcs5[int(r)])
        return np.array(codes_com, dtype=np.uint8)

