clear all;
close all;
clc;

base_path = 'F:\Diamond\Videos_ori';
save_path = 'F:\Diamond\Videos';

id_list = dir(fullfile(base_path,'*.mp4'));
id_exist = dir(fullfile(save_path,'*.mp4'));

fileids_exist = string(missing);
for idx = 1 : length(id_exist)
    % load
    fileids_exist(idx) = id_list(idx).name;
end
    

for idx = 1 : 600
    % load
    if  ~isempty(intersect(id_list(idx).name, fileids_exist))
        fprintf([id_list(idx).name, ' exist\n']);
        continue
    end
    
    video_read_name = fullfile(base_path, id_list(idx).name);
    v = VideoReader(video_read_name);
   
    % save
    video_save_name = fullfile(save_path, [id_list(idx).name(1:11), '.mp4']);
    v_save = VideoWriter(video_save_name, 'MPEG-4');
    open(v_save);
    
    fprintf(['\n', id_list(idx).name]);
    cnt = 1;
    while hasFrame(v)
        cnt_str = sprintf('%03d',cnt);
        fprintf([cnt_str,'/400...'])
 
        frame = readFrame(v);
        len = length(frame);  
        writeVideo(v_save,frame(:,1:len/2,:))

        fprintf('\b\b\b\b\b\b\b\b\b\b')
        cnt = cnt + 1;
    end
    close(v_save);
    
end


