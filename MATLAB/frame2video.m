%% Merge frames to one video

% clear all
close all
clc

src_path = 'D:\OneDrive - City University of Hong Kong\chow\track_result\921B\tmp_save\10375539730\all_frame_pre_processing';
save_path = 'D:\OneDrive - City University of Hong Kong\chow\track_result\921B\tmp_save\10375539730\';

diamond_id = '10375539730';
v_pred = VideoWriter(fullfile([save_path,diamond_id, '_pre_processing']),'MPEG-4');
open(v_pred);

for ii = 1:400
    idx = sprintf('%03d',ii)
    img = imread(fullfile(src_path,[diamond_id,'_',idx,'.png']));
    writeVideo(v_pred,img);
end
close(v_pred)