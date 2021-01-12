clear;close all;
%% settings
Sets = 'Set5'
folder = 'E:\Êý¾Ý¼¯\classical_SR_datasets\Set5'
savepath = [fullfile('..', Sets), '_mat'];

if ~exist(savepath, 'dir')
    mkdir(savepath);
end

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

scale = [2, 3, 4];

for i = 1 : length(filepaths)
    im = imread(fullfile(folder,filepaths(i).name));
    if size(im,3)>1
        im = rgb2ycbcr(im);
        im = im(:, :, 1);
    end
    for s = 1 : length(scale) 
        im_gt_y = modcrop(im, scale(s));
        im_gt_y = double(im_gt_y) / 255;
        im_l = imresize(im_gt_y,1/scale(s),'bicubic');
        im_b = imresize(im_l,scale(s),'bicubic');
        im_gt_y = im_gt_y * 255.0;
        im_l_y = im_l * 255.0;
        im_b_y = im_b * 255.0;
        last = length(filepaths(i).name)-4;
        filename = sprintf('%s/%s_x%s.mat', savepath, filepaths(i).name(1 : last),num2str(scale(s)));
        save(filename, 'im_gt_y', 'im_b_y', 'im_l_y');
    end
end
