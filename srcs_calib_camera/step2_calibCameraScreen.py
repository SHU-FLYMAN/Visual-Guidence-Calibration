from scipy.io import loadmat
from utils import load_config_screen
from Phaser import Phaser
from Calib import *


def calibScreen(config_file, gamma_file, data_folder, out_folder, calib_num, ends=".bmp", show=False):
    basefolder = data_folder.split("/")[-1]
    out_folder = os.path.join(out_folder, basefolder)
    phase_folder = os.path.join(out_folder, "phase")
    calib_folder = os.path.join(out_folder, "calib")
    # 01 读取标定配置
    config = load_config_screen(config_file)
    # 02 读取图片，进行标定
    rows = config["rows"]
    cols = config["cols"]
    pattern_size = (cols, rows)
    square_size = config["square_size"]
    # 03 建立世界坐标系
    pts3d_world = np.zeros((cols * rows, 3), np.float32)
    pts3d_world[:, :2] = np.mgrid[0: cols, 0: rows].T.reshape(-1, 2)
    pts3d_world *= square_size

    # 04 检测角点
    pts3d_world_all = []
    pts2d_chess_all = []
    pts2d_circle_all = []
    pts2d_phase_all, pts3d_phase_all = [], []
    pts2d_phase_all_iter, pts3d_phase_all_iter = [], []
    pts2d_phase_all_iter_p, pts3d_phase_all_iter_p = [], []  # 普通高斯迭代

    # 加载相位检测器
    phaser = Phaser()
    phaser.load_config(config["N"], config["n"], config["B_min"], config["iter_num"], gamma_file)
    image_size = None
    for idx in range(1, calib_num + 1):
        print("idx=", idx)
        # 01 检测棋盘格角点
        img_chess = cv2.imread(os.path.join(data_folder, str(idx), "1" + ends), 0)
        if image_size is None:
            image_size = img_chess.shape
        found_chess, pts2d_chess = detect_corners(img_chess, pattern_size, show)
        if found_chess:
            pts2d_chess_all.append(pts2d_chess)
            pts3d_world_all.append(pts3d_world)

        # 02 圆环检测
        img_circle = 255 - cv2.imread(os.path.join(data_folder, str(idx), "2" + ends), 0)
        found_circle, pts2d_circle = detect_circles(img_circle, pattern_size, show)
        if found_circle:
            pts2d_circle_all.append(pts2d_circle)

        # 03 相移法
        if flag_phase:
            pha_x, pha_y = None, None
            pha_x_iter, pha_y_iter = None, None

            os.makedirs(phase_folder , exist_ok=True)
            save_file_pha_x = os.path.join(phase_folder, "pha_x_" + str(idx) + ".mat")
            save_file_pha_y = os.path.join(phase_folder, "pha_y_" + str(idx) + ".mat")
            # 如果不重新计算，那么就加载数据
            if not flag_recalc:
                # 如果文件存在
                if os.path.exists(save_file_pha_x):
                    d_mat = loadmat(save_file_pha_x)
                    pha_x = d_mat["pha_absolute"]
                    pha_x_iter = d_mat["pha_absolute_iter"]
                    print("成功加载文件:", save_file_pha_x)
                if os.path.exists(save_file_pha_y):
                    d_mat = loadmat(save_file_pha_y)
                    pha_y = d_mat["pha_absolute"]
                    pha_y_iter = d_mat["pha_absolute_iter"]
                    print("成功加载文件:", save_file_pha_y)

            if (pha_x is None) or (pha_y is None) or (pha_x_iter is None) or (pha_y_iter is None):
                # 编码图片数量：相移图片+格雷码+互补格雷码+黑白图片
                N_max = 32
                start_idx = 2
                step = int(N_max / config["N"])
                files_x, files_y = [], []
                i = 0
                for i in range(start_idx + step, start_idx + N_max + step, step):
                    file_x = os.path.join(data_folder, str(idx), str(i) + ends)
                    files_x.append(file_x)
                # 互补格雷码: n + 互补 +黑白图片
                for g in range(config["n"] + 2 + 1):
                    file_x = os.path.join(data_folder, str(idx), str(i + g + 1) + ends)
                    files_x.append(file_x)
                x_num = N_max + config["n"] + 2 + 1
                idx_y = [x_num + int(os.path.basename(file).split(".")[0]) for file in files_x]
                files_y = [os.path.join(data_folder, str(idx), str(i) + ends) for i in idx_y]

                # 计算相位并且保存
                d_pha_x = phaser.calcAbsolutePhase(files_x)
                d_pha_y = phaser.calcAbsolutePhase(files_y)
                pha_x, pha_x_iter = d_pha_x["pha_absolute"], d_pha_x["pha_absolute_iter"]
                pha_y, pha_y_iter = d_pha_y["pha_absolute"], d_pha_y["pha_absolute_iter"]

                # 重新保存
                savemat(save_file_pha_x, d_pha_x)
                savemat(save_file_pha_y, d_pha_y)
                print("save pha_x file into:", save_file_pha_x)
                print("save pha_y file into:", save_file_pha_y)

            # 无迭代
            pts2d_phase, pts3d_phase = detect_phase_shift(pha_x, pha_y, pattern_size, pts2d_chess, config)

            # 迭代
            pts2d_phase_iter, pts3d_phase_iter = detect_phase_shift(
                pha_x_iter, pha_y_iter, pattern_size, pts2d_chess, config, False)

            # 迭代 + 插值
            pts2d_phase_iter_p, pts3d_phase_iter_p = detect_phase_shift(
                pha_x_iter, pha_y_iter, pattern_size, pts2d_chess, config, True)

            # 展示细化的相位
            if show:
                show_phase_target(pha_x_iter, pts2d_phase_iter, save_file_pha_x.replace(".mat", ".jpg"))
                show_phase_target(pha_y_iter, pts2d_phase_iter, save_file_pha_y.replace(".mat", ".jpg"))

                show_phase_target(pha_x_iter, pts2d_phase_iter_p, save_file_pha_x.replace(".mat", "_p.jpg"))
                show_phase_target(pha_y_iter, pts2d_phase_iter_p, save_file_pha_y.replace(".mat", "_p.jpg"))

            # 添加结果
            pts2d_phase_all.append(pts2d_phase)
            pts3d_phase_all.append(pts3d_phase)

            pts2d_phase_all_iter.append(pts2d_phase_iter)
            pts3d_phase_all_iter.append(pts3d_phase_iter)

            pts2d_phase_all_iter_p.append(pts2d_phase_iter_p)
            pts3d_phase_all_iter_p.append(pts3d_phase_iter_p)
            cv2.waitKey(10)
            cv2.destroyAllWindows()
    # 标定次数 >=3
    if calib_num >= 3:
        # 05 相机标定
        # 棋盘格（快速标定）
        calib_result_chess = calib_camera(
            "chess", pts3d_world_all, pts2d_chess_all, image_size, config["flag_calib"], calib_folder)

        # 圆环（对比方法）
        calib_result_circle = calib_camera(
            "circle", pts3d_world_all, pts2d_circle_all, image_size, config["flag_calib"], calib_folder)

        if flag_phase:
            # 普通方法
            calib_result_phase = calib_camera(
                "phase", pts3d_phase_all, pts2d_phase_all, image_size, config["flag_calib"], calib_folder)
            # 高斯迭代
            calib_result_phase_iter = calib_camera(
                "phase-iter", pts3d_phase_all_iter, pts2d_phase_all_iter, image_size, config["flag_calib"], calib_folder)
            calib_result_phase_iter_inter = calib_camera(
                "phase-iter_p", pts3d_phase_all_iter_p, pts2d_phase_all_iter_p, image_size, config["flag_calib"], calib_folder)


if __name__ == '__main__':
    flag_phase  = True                       # whether to calculate the phase-shift target
    flag_recalc = False                      # whether to use the previous phase maps
    calib_num   = 8                          # number of calibration images
    config_file = "data/config_screen.xml"   # Config of chessboard
    gamma_file  = "../srcs_gamma/gamma.mat"  # gamma parameter
    data_folder = "E:/calib/data/d3-free"    # data folder
    out_folder  = "out/calib"                # output folder
    calibScreen(config_file, gamma_file, data_folder, out_folder, calib_num, show=True)
