% clear all
close all
clc

% 1=save 1st frame for all videos, 2=save all frames for one video
option = 2;  

% path
base_path = '/media/hdd/css_data/hdr_data_v2/20200205 20 HDR images and videos (81 - 100)';
save_path = '/home/lingjia/Documents/diamond_data/frames/80_100';
videoPath  = dir(fullfile(base_path, '*.mp4'));
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

switch option
    case 1
        % save 1st frame for all videos
        for ii = 1:length(videoPath)
            v = VideoReader(fullfile(base_path, videoPath(ii).name));
%             frame = readFrame(v);
            frame = read(v, 400);
            frame_name = fullfile(save_path,[videoPath(ii).name(1:11),'.png']);
            fprintf([videoPath(ii).name,'\n']);
            len = length(frame);
            imwrite(frame(:,1:len/2,:),frame_name);
        end

    case 2
        % save 400 frame for 1 video
%         video_filename = '10375539730(G)-R1-Darkfield-02.mp4';
        video_filename = videoPath(1).name;
        file_id = video_filename(1:11);
        
        save_path = fullfile(save_path, [file_id, '_all']);
        if ~exist(save_path, 'dir')
            mkdir(save_path)
        end

        v = VideoReader(fullfile(base_path, video_filename));
        cnt = 1;
        while hasFrame(v)
            frame = readFrame(v);
            cnt_str = sprintf('%03d',cnt);
            fprintf([cnt_str,'/400...','\n'])
            frame_name = fullfile(save_path, [file_id,'_',cnt_str,'.png']);
            len = length(frame);
            imwrite(frame(:,1:len/2,:), frame_name)
            cnt = cnt + 1;
        end
end



