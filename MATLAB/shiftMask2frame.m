clear all;
close all;
clc;

%% Path
path_mask = 'F:\Diamond\Masks';
path_img = 'F:\Diamond\Images';
path_frame = 'F:\Diamond\1stFrame';
path_save = 'F:\Diamond\frame_mask\mask_new_591';
if ~exist(path_save, 'dir')
   mkdir(path_save)
end

%% id_list
path = 'F:\Diamond\hdr_data_v2\20200506 60 HDR images and videos (531 - 590)';
input = struct([]);
id_list = searchSubdir(path, input);


%% id_exist & id_mask
path_exist1 = 'F:\Diamond\trajectory_1213\previous\all_reflection';
path_exist2 = 'F:\Diamond\trajectory_1213\previous\all_inclusion';
id_exist1 = dir(fullfile(path_exist1,'*.txt'));
id_exist2 = dir(fullfile(path_exist2,'*.txt'));

fileids_exist = string(missing);
for idx = 1 : length(id_exist1)
    fileids_exist(idx) = id_exist1(idx).name(1:11);
end
for idx = 1+length(id_exist1) : length(id_exist2)+length(id_exist1)
    fileids_exist(idx) = id_exist2(idx-length(id_exist1)).name(1:11);
end

id_mask = dir(fullfile(path_mask,'*_mask.png'));
fileids_mask = string(missing);
for idx = 1 : length(id_mask)
    fileids_mask(idx) = id_mask(idx).name(1:11);
end


%% Loop for images
fprintf(num2str(length(id_list)));
for idx = 49: length(id_list)
    id = id_list(idx).name(1:11);
    
    %% if exist / without mask
    if  ~isempty(intersect(id, fileids_exist))
        fprintf([id, ' exist\n']);
        continue
    end
    if  isempty(intersect(id, fileids_mask))
        fprintf([id, ' do not have inclusion\n']);
        continue
    end
    
    fprintf([num2str(idx),' ', num2str(id)]);
    
    v1_frame = imread(fullfile(path_frame,[id,'_001.png']));
    v1_img = imread(fullfile(path_img,[id,'.png']));
    v1_mask = imread(fullfile(path_mask,[id,'_mask.png']));

    v1_img = im2double(v1_img);
    v1_frame = im2double(v1_frame);
    v1_mask = im2double(v1_mask);
    
    mask_new = zeros(size(v1_mask));
    range_hw = 15;
    
    %% Gradient %%
    [ix,iy] = gradient(v1_img);
    igrad = abs(ix)+abs(iy);
    
    [fx,fy] = gradient(v1_frame);
    fgrad = abs(fx)+abs(fy);

    %% reflection
%     mask_single = v1_mask(:,:,1);
%     igrad_tmp = igrad.*mask_single;
%     % figure; imshow(imfuse(igrad,igrad_tmp));
%     
% %     filename = 'D:\OneDrive - City University of Hong Kong\1214\vis_mask_grad_enhance.png';
% %     frame = getframe(gca); 
% %     img = frame2im(frame); 
% %     imwrite(img,filename); 
%     
%     dist = zeros(1,3);
%     index = 0;
%     % neg = move up, pos = down, neg = move left, pos = right
%     %%%%%%%%%%%%%%% Loop for search window %%%%%%%%%%%%%%
%     for hh = -5:1:5
%         for ww = -5:1:5
%             index = index + 1;
% 
%             % MSE of pixel value %
% %                 frame_shift = circshift(v1_frame, [hh,ww]);
% %                 masked_frame = frame_shift.*mask_single;
% %                 figure; imshow(imfuse(frame_shift,mask_single));
% %                 mse = sum((masked_frame - masked_img).^2,'all');
% 
%             % Gradient %
%             fgrad_shift = circshift(fgrad, [hh,ww]);
%             fgrad_tmp = fgrad_shift.*mask_single;
% %             figure; imshow(imfuse(fgrad,mask_single));
%             mse_grad = sum((igrad_tmp - fgrad_tmp).^2,'all');
% 
%             dist(index,:)=[hh,ww,mse_grad];
%         end
%     end
%     [~,I] = min(dist(:,3));
%     info = dist(I,:);
%     sh = info(1); sw = info(2);
%     mask_shift = circshift(mask_single,[-sh,-sw]);
%     mask_new(:,:,1) = mask_shift;
% %     figure; imshow(imfuse(v1_frame, mask_new(:,:,1)));
%     fprintf(' Reflection');
    
    %% Inclusion
    mask_single = v1_mask(:,:,2);
    
    % Image
%     masked_img = v1_img.*mask_single;
%     figure; imshow(imfuse(v1_img, mask_single));

    % Gradient
    igrad_tmp = igrad.*mask_single;
    
    dist = zeros(1,3);
    index = 0;
    %%%%%%%%%%%%%%% Loop for search window %%%%%%%%%%%%%%
    for hh = -range_hw:1:range_hw
        for ww = -range_hw:1:range_hw
            index = index + 1;

           % Image %
%             frame_shift = circshift(v1_frame, [hh,ww]);
%             masked_frame = frame_shift.*mask_single;
%             figure; imshow(imfuse(frame_shift,mask_single));
%             mse = sum((masked_frame - masked_img).^2,'all');

            % Gradient %
            fgrad_shift = circshift(fgrad, [hh,ww]);
            fgrad_tmp = fgrad_shift.*mask_single;
%             figure; imshow(imfuse(fgrad,mask_single));
            mse_grad = sum((igrad_tmp - fgrad_tmp).^2,'all');

            dist(index,:)=[hh,ww,mse_grad];
        end
    end
    [~,I] = min(dist(:,3));
    info = dist(I,:);
    sh = info(1); sw = info(2);
    mask_shift = circshift(mask_single,[-sh,-sw]);
%     figure; imshow(imfuse(v1_frame,mask_shift));
%     figure; imshow(imfuse(v1_frame,mask_single));
    mask_new(:,:,2) = mask_shift;
    fprintf(' Inclusion');
    
    %% Save
    save_name = fullfile(path_save,[num2str(id),'_mask.png']);
    imwrite(mask_new,save_name);
    fprintf([' save...','\n'])
end




function y = searchSubdir(path,input)
    fprintf('%s\n', path)
    y = find_id_list(path, input);
    subdirs = dir(path);
    for jj = 1:length(subdirs)
        if isequal(subdirs(jj).name, '.') || isequal(subdirs(jj).name, '..') || ~subdirs(jj).isdir
            continue;
        end
        tmp_path = fullfile(path,subdirs(jj).name);
        y = searchSubdir(tmp_path, y);
    end
end

function y = find_id_list(data_path, input_patch)
    videoPath = dir(fullfile(data_path,'*.mp4'));
    if ~isempty(videoPath)
        if isempty(input_patch)
            y = videoPath;
        else
            y = [input_patch; videoPath];
        end
    else
        y = input_patch;
    end
end



