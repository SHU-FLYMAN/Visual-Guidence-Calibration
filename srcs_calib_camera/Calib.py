import os
import cv2
import numpy as np
from scipy.io import savemat
from matplotlib import pyplot as plt


paras = cv2.SimpleBlobDetector_Params()
paras.maxArea = 10e4
paras.minArea = 20
paras.filterByArea = True
blobDetector = cv2.SimpleBlobDetector_create(paras)


def detect_corners(img, pattern_size, show=False):
    found, pts2d = cv2.findChessboardCorners(img, pattern_size, None)
    if found:
        pts2d = cv2.cornerSubPix(img, pts2d, (11, 11), (-1, -1),
           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    else:
        raise FileNotFoundError("Never detected chessboard")
    if show:
        img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(img_show, pattern_size, pts2d, found)
        f = 0.4
        img_show = cv2.resize(img_show, None, fx=f, fy=f)
        cv2.imshow("chessboard", img_show)
        cv2.waitKey(10)
    return found, pts2d


def detect_circles(img, pattern_size, show=False):
    found, pts2d = cv2.findCirclesGrid(
        img, pattern_size, blobDetector=blobDetector,
        flags=cv2.CALIB_CB_SYMMETRIC_GRID | cv2.CALIB_CB_CLUSTERING)
    if not found:
        raise FileNotFoundError("Never detected circles in image.")
    if show:
        img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.drawChessboardCorners(img_show, pattern_size, pts2d, found)
        f = 0.4
        img_show = cv2.resize(img_show, None, fx=f, fy=f)
        cv2.imshow("circle", img_show)
        cv2.waitKey(10)
    return found, pts2d


def calc_pixel_pose(pha_x, pha_y, x, y, config):
    screen_pixel_x = pha_x[y, x] * config["Screen_Width_Pixel"]
    screen_pixel_y = pha_y[y, x] * config["Screen_Height_Pixel"]
    pt_world_x = screen_pixel_x * config["p"]
    pt_world_y = screen_pixel_y * config["p"]
    return pt_world_x, pt_world_y


# 找到跟棋盘格角点接近的值，然后计算世界坐标，
def detect_phase_shift(pha_x, pha_y, pattern_size, pts2d_pixels, config, inter=False):
    # 01 计算理想角点的相位值
    cols, rows = pattern_size

    # 02 寻找准确的相位值点
    pts2d_pixels_int = pts2d_pixels.astype(np.int32)
    if inter:
        p2d_all = []
        num = 3  # 分为3段
        for row in range(rows - 1):
            for col in range(cols - 1):
                idx_p1 = row * cols + col
                idx_p2 = idx_p1 + 1
                idx_p3 = (row + 1) * cols + col
                idx_p4 = idx_p3 + 1
                p1 = pts2d_pixels_int[idx_p1]
                p2 = pts2d_pixels_int[idx_p2]
                p3 = pts2d_pixels_int[idx_p3]
                p4 = pts2d_pixels_int[idx_p4]
                p13s, p12s, p24s, p34s = [], [], [], []
                for i in range(1, num):
                    p13s.append(p1 + (p3 - p1) / num * i)
                    p12s.append(p1 + (p2 - p1) / num * i)
                    p24s.append(p2 + (p4 - p2) / num * i)
                    p34s.append(p3 + (p4 - p3) / num * i)
                ps = []
                for i in range(1, num):  # 行
                    p_l, p_r = p13s[i - 1], p24s[i - 1]
                    for j in range(1, num):  # 列
                        p = p_l + (p_r - p_l) / num * j
                        ps.append(p)
                ps.extend([p1, p2, p3, p4])
                ps.extend(p13s)
                ps.extend(p12s)
                ps.extend(p24s)
                ps.extend(p34s)
                p2d_all.extend(ps)
        p2d_all = np.array(p2d_all, dtype=np.int32)
    else:
        p2d_all = pts2d_pixels_int

    # 转化为世界坐标系
    all_num = p2d_all.shape[0]
    p3d_all = np.zeros([all_num, 3], dtype=np.float32)
    for i in range(all_num):
        p_2d = np.squeeze(p2d_all[i, :])
        X, Y = calc_pixel_pose(pha_x, pha_y, p_2d[0], p_2d[1], config)
        p3d_all[i, :] = np.array([X, Y, 0], np.float32)
    p2d_all = p2d_all.astype(np.float32)
    return p2d_all, p3d_all


# matlab跟opencv的重投影计算方式有一些区别
# MSE Error: https://blog.csdn.net/qq_32998593/article/details/113063216
def reproject_error(pts3d, pts2d, rvecs, tvecs, mtx, dist):
    num = len(pts2d)
    dxs, dys = [], []
    for i in range(num):
        pts2d_1 = np.squeeze(pts2d[i])
        # 重投影点
        pts2d_2, _ = cv2.projectPoints(pts3d[i], rvecs[i], tvecs[i], mtx, dist)
        pts2d_2 = np.squeeze(pts2d_2)
        # 计算dx, dy
        d = pts2d_2 - pts2d_1
        dx, dy = d[:, 0], d[:, 1]
        dxs.append(dx)
        dys.append(dy)
    return dxs, dys


def calib_camera(name, pts3d_all, pts2d_all, image_size, flags, folder):
    error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        pts3d_all,
        pts2d_all,
        image_size,
        None,
        None,
        flags=flags
    )
    dxs, dys = reproject_error(pts3d_all, pts2d_all, rvecs, tvecs, mtx, dist)

    ds = np.sqrt(np.power(dxs, 2) + np.power(dys, 2))
    error_l2s = np.mean(ds, axis=1)
    # 平均公式距离
    error_l2 = np.mean(ds)
    # 展示标定结果
    calib_result = {
        "name": name,
        "error": error,
        "error_l2": error_l2,
        "error_l2s": error_l2s,
        "dxs": dxs,
        "dys": dys,
        "ds": ds,
        "camera_matrix": mtx,
        "dist": dist,
        "rvecs": rvecs,
        "tvecs": tvecs
    }
    print("\n", name)
    print("error:", error)
    print("error_l2:", error_l2)
    print("camera_matrix", mtx)
    print("dist", dist)

    os.makedirs(folder, exist_ok=True)
    save_file = os.path.join(folder, name + "_" + str(len(tvecs)) + ".mat")
    savemat(save_file, calib_result)
    print("保存相机参数到:", name, " path=", save_file, "\n")
    return calib_result


def calib_camera_comp(name, pts3d_all, pts2d_all, image_size, flags, folder):
    print("误差校正标定")
    # 01 初步标定
    error, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        pts3d_all,
        pts2d_all,
        image_size,
        None,
        None,
        flags=flags
    )
    # 补偿折射误差
    num = len(pts2d_all)
    d = 1  # mm
    nt = 1.5
    pts3d_all_right = []
    h_2, w_2 = image_size[0] / 2, image_size[1] / 2
    n = np.array([0, 0, 1.], dtype=np.float64)
    for i in range(num):
        pts3d_right = np.zeros_like(pts3d_all[i])
        pts3d = np.squeeze(pts3d_all[i])
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i]

        for idx in range(len(pts3d)):
            # x, y, z
            QC = np.squeeze(np.matmul(-R.T, t) - t) - pts3d[idx]
            # 单位是弧度制pi
            alpha = np.arccos(np.dot(QC, n) / (np.sqrt(np.dot(QC, QC) * np.dot(n, n))))
            alpha_rad = alpha / np.pi * 180
            # 长度
            SS_length = np.abs(d * (1 - 1 / nt) * np.sin(alpha))  # alpha的正负角度不影响计算
            # 投影
            SS_direct = QC + np.dot(np.dot(QC, n), n)
            alpha2 = np.arctan2(SS_direct[0], SS_direct[1])
            alpha2_rad = alpha2 / np.pi * 180
            dx = np.abs(SS_length * np.sin(alpha2))
            dy = np.abs(SS_length * np.cos(alpha2))
            X = pts3d[idx][0]
            Y = pts3d[idx][1]
            if pts2d_all[i][idx][0][0] - w_2 >= 0:
                p_x = 1
            else:
                p_x = -1
            if pts2d_all[i][idx][0][1] - h_2 >= 0:
                p_y = 1
            else:
                p_y = -1
            pts3d_right[idx, 0] = X + dx * p_x
            pts3d_right[idx, 1] = Y + dy * p_y
        pts3d_all_right.append(pts3d_right)

    # 调用之前API标定相机
    calib_result = calib_camera(name, pts3d_all_right, pts2d_all, image_size, flags, folder)
    return calib_result


def show_phase_target(pha, pts2d, name):
    h, w = pha.shape
    scale = 0.5
    pts2d_s = np.copy(pts2d)
    pts2d_s *= scale
    pha_s = np.copy(pha)
    pha_s = cv2.resize(pha_s, (int(h * scale), int(w * scale)))
    pha_s = np.nan_to_num(pha_s, nan=0)
    pha_s = (pha_s * 220).astype(np.uint8)
    pha_s = cv2.cvtColor(pha_s, cv2.COLOR_GRAY2BGR)
    circle_c = (0, 0, 255)
    num = pts2d_s.shape[0]
    for i in range(num):
        x = pts2d_s[i, :, 0]
        y = pts2d_s[i, :, 1]
        cv2.circle(pha_s, (x, y), 3, circle_c, -1)
    cv2.imshow(name, pha_s)
    cv2.imwrite(name, pha_s)
    cv2.waitKey(10)
