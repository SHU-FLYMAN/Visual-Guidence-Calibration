%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The Matlab implementation of the paper:                        %
% "Calibration Wizard: A Guidance System for Camera Calibration  %
% Based on Modelling Geometric and Corner Uncertainty" ICCV 2019 %
% Songyou Peng, Peter Sturm                                      %
%                                                                %
% The code can only be used for research purposes.               %
%                                                                %
% Copyright (C) 2019 Songyou Peng                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear;
close all;


%% 01 设置参数
calib_num           = 3;                   % the index of the calibration   
name                = "phase-iter_inter";  % the name of the calibration
config_file         = '../srcs_calib_camera/data/config_screen.xml';  % config file 
calib_result_folder = "../srcs_calib_camera/out/calib/";       % calibration folder
% calibration result file, it will used to calculate the next best pose.
calib_result_file   = calib_result_folder + name + "_" + int2str(calib_num) + ".mat";
% the output file that storages the infomation of the next best pose.
next_pose_file      =  "./out/nextpose_points_" + int2str(calib_num + 1) + ".txt";


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
autoCorr_flag = 1;         % 1: consider autocorretion matrix  0: otherise
dist_border = 5;           % smallest distance between the next pose and border (in pixel)
dist_neighbor = 5;         % smallest distance of neighboring corner points (in pixel)
tranlation_bound = 200;    % bound of translation in optimization, may need to change for different cameras
% display                   %
optim_display = 'final';    % level of display. 'iter'|'final'|'off'
plot_uncertainty = 1;       % 1: display uncertainty map  0: not display
pixel_gap = 32;             % calculate uncertainty every pixel_gap pixels, e.g. every 10 pixels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 02 Extract the calibration infomation of initial calibration process
[basicInfo, intrinsicPara, extrinsicPara] = extract_info(config_file, calib_result_file);
basicInfo.dist_border   = dist_border;
basicInfo.dist_neighbor = dist_neighbor;

board_Width  = basicInfo.board_Width;  % 标定板宽度
board_Height = basicInfo.board_Height; % 标定板高度
image_Width  = basicInfo.image_Width;  % 图像宽度
image_Height = basicInfo.image_Height; % 图像高度
square_Size  = basicInfo.square_Size;  % 网格大小
num_frame    = basicInfo.num_frame;    % 图片数量
rot_Mat      = extrinsicPara.rot_Mat;  % 旋转矩阵
t_Vec        = extrinsicPara.t_Vec;    % 平移矩阵


%% 03 定义角点的世界坐标系点
% TODO 修改Square_Size
corners_world = zeros(board_Width * board_Height, 3);
for i = 1 : board_Height
    for j = 1 : board_Width
        corners_world(j + (i - 1) * board_Width, 1) = (j - 1) * square_Size;
        corners_world(j + (i - 1) * board_Width, 2) = (i - 1) * square_Size;
    end
end
corners_world = corners_world';

%% 04 世界坐标系转换到相机坐标系
% S = R * Q + t
corners_camera = zeros(3, board_Width * board_Height, num_frame);
for m = 1 : num_frame
    for i = 1 : board_Height
        for j = 1 : board_Width
            corners_camera(:,j + (i - 1) * board_Width,m) = rot_Mat(:,:,m) * corners_world(:,j + (i - 1) * board_Width) + t_Vec(:,m);
        end
    end
end

%% 05 计算Jacobian矩阵 J = [A,B]
[A,B] = build_Jacobian(intrinsicPara, extrinsicPara, basicInfo, corners_camera, corners_world);

%% 06 构建角点矩阵
ACMat = eye(size(A,1));
if autoCorr_flag == 1
    ACMat = buildAutoCorrMatrix(corners_camera, intrinsicPara, basicInfo);
end

%% 07 显示不确定性图
if plot_uncertainty
    num_intr = length(fieldnames(intrinsicPara));
    J = [A, B];
    M = J'*ACMat*J;
    U = M(1:num_intr,1:num_intr);
    W = M(1:num_intr,num_intr+1:end);
    V = M(num_intr+1:end,num_intr+1:end);
    Sigma = inv(U- W*inv(V)*W');
    uncertainty_map(Sigma, intrinsicPara, basicInfo, pixel_gap);
end

%% 08 优化求最优位姿 Numerically get the next pose using optimizer (Local or Global(SA))

% 设置外参的初始参数 Set initial extrinsic parameters
x = [0, 0, 0, mean(t_Vec(1,:)), mean(t_Vec(2,:)), mean(t_Vec(3,:))];

% 设置全局优化方法 Global optimization method
tic;
lb = [0, 0, 0, -tranlation_bound, -tranlation_bound, 0]; % lower bound
ub = [pi/6, pi/12, pi/6, tranlation_bound, tranlation_bound, mean(t_Vec(3,:))]; % upper bound

options = saoptimset('Display', optim_display); % check other options for SA in https://www.mathworks.com/help/gads/saoptimset.html
if autoCorr_flag == 1
    ACMat_extend = zeros(size(ACMat) + [2*board_Height*board_Width, 2*board_Height*board_Width]);
    ACMat_extend(1:size(ACMat,1), 1:size(ACMat,2)) = ACMat;
    [x,fval,exitFlag,output] = simulannealbnd(@(x)cost_function(x, A, B, corners_world, intrinsicPara, basicInfo, ACMat_extend), x, lb, ub, options);
else
    [x,fval,exitFlag,output] = simulannealbnd(@(x)cost_function(x, A, B, corners_world, intrinsicPara, basicInfo), x, lb, ub, options);
end
toc;


%% 09 绘制新的位姿
P_next = compute_nextpose_points(x, corners_world, intrinsicPara, basicInfo);
plot_nextpose(P_next, basicInfo);

writematrix(P_next, next_pose_file);

%% 10 估计新姿势后的预期不确定度图
if plot_uncertainty
    [A_new, B_new] = build_Jacobian_nextpose(intrinsicPara, basicInfo, corners_world, x);

    A = [A;A_new];
    B = [B, zeros(size(B,1),6);zeros(2*board_Width*board_Height,size(B,2)), B_new];
    J = [A,B];
  
    if autoCorr_flag == 1
        [ACMat_new,~] = buildSingleAutoCorrMatrix(P_next, basicInfo);
        ACMat_extend = zeros(size(ACMat) + size(ACMat_new));
        ACMat_extend(1:size(ACMat,1), 1:size(ACMat,2)) = ACMat;
        ACMat_extend (end-size(ACMat_new,1)+1 : end, end-size(ACMat_new,1)+1 : end) = ACMat_new;
    else
        ACMat_extend = eye(size(J,1));
    end

    M = J' * ACMat_extend * J;
    U = M(1:num_intr,1:num_intr);
    W = M(1:num_intr,num_intr+1:end);
    V = M(num_intr+1:end,num_intr+1:end);
    F = inv(U- W*inv(V)*W');

    uncertainty_map(F, intrinsicPara, basicInfo, pixel_gap);
end