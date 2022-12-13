function [basicInfo, intrinsicPara, extrinsicPara] = extract_info(config_file, calib_result_file)
%% 01 标定配置
Info = xmlread(config_file);
image_Width  = load_node(Info, 'image_Width');   % 图像的宽
image_Height = load_node(Info, 'image_Height');  % 图像的高
board_Width  = load_node(Info, 'BoardSize_Width');   % 标定板宽度
board_Height = load_node(Info, "BoardSize_Height");  % 标定板高度
Screen_Width_Pixel = load_node(Info, "Screen_Width_Pixel");
inch = load_node(Info, "inch");
Sceen_Size = load_node(Info, "Screen_Size");
Screen_Width_Ratio = load_node(Info, "Screen_Width_Ratio");
Screen_Height_Ratio = load_node(Info, "Screen_Height_Ratio");

Screen_Width = Screen_Width_Ratio / sqrt(Screen_Width_Ratio ^ 2 + Screen_Height_Ratio ^ 2) * inch * Sceen_Size;
p = Screen_Width / Screen_Width_Pixel;  % 单个像素大小

square_size_pixel = Screen_Width_Pixel / (board_Width + 3);
square_Size  = square_size_pixel * p;   % 棋盘格尺寸

basicInfo.board_Width  = board_Width;
basicInfo.board_Height = board_Height;
basicInfo.image_Width  = image_Width;
basicInfo.image_Height = image_Height;
basicInfo.square_Size  = square_Size;

%% 02 相机内参
p = load(calib_result_file);          
cameraMatrix = p.camera_matrix;
% 这里fx=fy，由于设置的问题
intrinsicPara.f = cameraMatrix(1,1);
intrinsicPara.u = cameraMatrix(1,3);
intrinsicPara.v = cameraMatrix(2,3);


%% 03 畸变系数
k1_opt = load_node(Info, "Calibrate_AssumeZerok1Distortion");
k2_opt = load_node(Info, "Calibrate_AssumeZerok2Distortion");

% 如果为真（非零），则径向畸变的 k1 设置为零
% 如果为真（非零），则径向畸变的 k2 设置为零
if k1_opt ~= 0
    if k2_opt ~= 0
        dist_type = 'no_dist';
    end
else
    if k2_opt ~= 0
        dist_type = 'radial1';
    else
        dist_type = 'radial2';
    end
end

% 目前只支持K1,K2
dist_coeff = p.dist;
switch dist_type
    case 'no_dist'
        % Does not have distortion parameters
    case 'radial1'
        intrinsicPara.k1 = dist_coeff(1); 
    case 'radial2'
        intrinsicPara.k1 = dist_coeff(1);
        intrinsicPara.k2 = dist_coeff(2);    
    otherwise
        disp('No suitable type');
        return;
end

%% 04 外参R、t
raw_rotation = p.rvecs;
num_frame = size(raw_rotation, 1);
rot_Mat = zeros(3, 3, num_frame);
for i = 1: num_frame
    r_vec = raw_rotation(i, :);
    r_mat = rotationVectorToMatrix(r_vec);
    rot_Mat(:, :, i) = r_mat';
end

extrinsicPara.rot_Mat = rot_Mat;
extrinsicPara.t_Vec = p.tvecs';

basicInfo.num_frame = num_frame; % 帧数
end


% 读取节点数据的函数
function [info] = load_node(file, name)
node = file.getElementsByTagName(name).item(0);
info = str2double(node.getTextContent);
end