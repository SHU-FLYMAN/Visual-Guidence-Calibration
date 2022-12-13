close all;
clear; clc;
tic;

%% 01 Hyperparameters
WIDTH  = 2048;  % width pixel of screen
HEIGHT = 1536;  % height pixel of screen
% Not all grays of ststen are conform to the gamma curve
start  = 40;    % minimum gray level to fit
ends   = 220;   % maximum gray level to fit
step   = 1;     % step between grays
folder = "patterns";  % save folder for patterns
suffix = "bmp";       

%% 02 Generate gamma patterns
mkdir(folder);
img = ones(HEIGHT, WIDTH, 'uint8');
disp("writing...");
for g = start: step: ends
    img_g = img * g;
    file = folder + "/" + int2str(g) + "." + suffix;
    imwrite(img_g, file);
    disp(file);
end

toc;
