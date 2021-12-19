%% Delete crust (black border and white aperture) for video frames

base_path = 'D:\OneDrive - City University of Hong Kong\css\data\IG\first_frame0\';
save_path = 'D:\OneDrive - City University of Hong Kong\css\data\IG\first_frame\';
img_list = dir([base_path,'\', '*.png']);
% frame_name = '10376463002.png';


for idx = 1:length(img_list)
    frame_name = img_list(idx).name;
    save_name = [save_path,frame_name];
    f = imread([base_path,frame_name]);
%     figure; imshow(f)
    f = f(:,:,3);

    [h,w] = size(f);
    center_h = h/2+3;
    center_w = w/2+7;
    
    [aa,bb] = find(f(:,600)==255);
    if length(aa)>=3
        if aa(3)<80 || aa(3)>600
            start = 80;
        else
            start = aa(3);
        end
    end

    r1 = regionGrow(f,start,w/2);
%     figure;imshow(r1)
    r2 = regionGrow(r1,h/2,w/2);
%     figure;imshow(r2)
    dist_h = 0;
    dist_w = 0;
    for ii=1:h
        for jj=1:w
            if r2(ii,jj)>=1
                dist_h(ii,jj) = abs(center_h-ii);
                dist_w(ii,jj) = abs(center_w-jj);
            else
                dist_h(ii,jj)=0;
                dist_w(ii,jj)=0;
            end
        end
    end
    rh = max(dist_h(:));
    rw = max(dist_w(:));
    
    rr = ones(h,w);
    for ii=1:h
        for jj=1:w
            if ((center_h-ii)/(rh+2))^2+((center_w-jj)/(rw+2))^2<1
                rr(ii,jj) = 0;
            end
        end
    end
%     figure; imshow(rr)
    rr = repmat(rr,[1 1 3]);

    f = imread([base_path,frame_name]);
    f = im2double(f);
    new_f = f;
    new_f(rr>0) = 1;
%     imshow([f,new_f])
    imwrite(new_f,save_name);

    fprintf([frame_name ' ' num2str(start)]);
    fprintf('\n')
end
