close all;
clear; clc;
tic;

%% 01 Hyperparameters
start  = 40;
ends   = 220;
step   = 1;
folder = "data";
suffix = "bmp";
save_file = "gamma.mat";

%% 02 确定图像的轮廓
img = imread(folder + "/" + int2str(220) + "." + suffix);
str = '1. Click to select initial contour location. 2. Double-click to confirm and proceed.';
figure(); imshow(img); title(str,'Color','b','FontSize',12);
fprintf('\nNote: Click close to object boundaries for more accurate result.\n')

% 交互式地指定初始轮廓：鼠标左键每单击一处确定选取轮廓多边形一个顶点，双击完成选取
mask = roipoly; % 用鼠标画多边形的函数
figure; imshow(mask); title('MASK');
pixel_num = sum(sum(mask));
mask = uint8(mask);  % 用于后续计算

%% 03 获得实际灰度值
grays = start: step: ends;
[~, num] = size(grays);
grays_actual = zeros(1, num);
disp("reading images...");
idx = 1;
for g = grays
    file = folder + "/" + int2str(g) + "." + suffix;
    disp(file);
    img = imread(file);
    img = double(img .* mask);
    grays_actual(1, idx) = sum(sum(img)) / pixel_num;
    idx = idx + 1;
end

%% 04 计算伽马曲线 y = a x^b + c  // b：gamma值
[a, b, c] = fit_gamma_curve(grays, grays_actual);

grays_correct = gamma_correct(grays_actual, a, b, c);

save(save_file, "b","a", "c"); disp(strcat("保存伽马到文件：", save_file));

%% 05 查看gamma校正结果
figure(); 
hold on;
plot(grays, grays, "--");
plot(grays, grays_actual);
plot(grays, grays_correct);
axis([0 260,0,260])
xlabel("理想灰度");
ylabel("实际灰度");
legend("理想", "实际", "校正", 'Location','West');
title("gamma校正结果")

% 误差
figure(); plot(grays, grays_correct - grays); title("误差水平");

toc;

% 程序：拟合gamma曲线 y = a x^b + c
function [a, b, c] = fit_gamma_curve(grays, grays_actual)
% grays、gyras_actual：0-255
grays = grays / 255.;
grays_actual = grays_actual / 255.;

%% Fit: '伽马曲线'.
[xData, yData] = prepareCurveData(grays, grays_actual);

% Set up fittype and options.
ft = fittype( 'power2' );
opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
opts.Display = 'Off';
% a, b,c 初始值
opts.StartPoint = [1 1 0];
% Fit model to data.
[fitresult, ~] = fit( xData, yData, ft, opts );
a = fitresult.a;
b = fitresult.b;
c = fitresult.c;
end

% 程序，返回理想灰度
function [grays_correct] = gamma_correct(grays_actual, a, b, c)
grays_actual = grays_actual / 255.;
grays_correct = (((grays_actual - c) / a) .^ (1 / b)) * 255.;
end
