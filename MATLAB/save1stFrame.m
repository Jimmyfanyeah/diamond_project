clc;
clear all;

% path
base_path = 'F:\diamond_project\hdr_data_v2';
save_path = 'F:\diamond_project\1stFrame';
if ~exist(save_path, 'dir')
   mkdir(save_path)
end

y = searchSubdir(base_path,save_path);

function y = searchSubdir(path,save_path)
    fprintf([path,'\n']);
    y = save1stFrame_path(path,save_path);
    subdirs = dir(path);
    for jj = 1:length(subdirs)
        if isequal(subdirs(jj).name, '.') || isequal(subdirs(jj).name, '..') || ~subdirs(jj).isdir
            continue;
        end
        tmp_path = fullfile(path,subdirs(jj).name);
        y = searchSubdir(tmp_path,save_path);
    end
end

function y = save1stFrame_path(data_path, save_path)
    videoPath = dir(fullfile(data_path,'*.mp4'));
    for ii = 1 : length(videoPath)
        v = VideoReader(fullfile(data_path, videoPath(ii).name));
        frame = readFrame(v);
    %     frame = read(v, 400);
        frame_name = fullfile(save_path,[videoPath(ii).name(1:11),'_001.png']);
        fprintf([videoPath(ii).name,'\n']);
        len = length(frame);
        imwrite(frame(:,1:len/2,:),frame_name);
    end
    y = 1;
end
