import os
import numpy as np 
from PIL import Image 
from skimage.measure import label, regionprops
from math import ceil

# src_folder = "/Users/lingjia/Chow_Proj/processed_data_clipped"
# save_folder = "/Users/lingjia/Chow_Proj/processed_data_new_clipped_intersect"
# tar_size = (400, 400)
# stride = 200

def handle_image(filename,label_filename,src_folder,save_folder,tar_size=(400,400),stride=200):
    img = Image.open(os.path.join(src_folder, filename))
    H, W = img.size 
    if img.mode != 'RGB':
        img = img.convert('RGB')

    for ii in range(int((H-tar_size[0])/stride)+1):
        print(ii)
        for jj in range(int((W-tar_size[1])/stride)+1):
            rect = (jj*stride, ii*stride, jj*stride+tar_size[1], ii*stride+tar_size[0])
            tmp_name = filename[:11]+'_'+str(ii)+'_'+str(jj)

            try:
                mask_img = Image.open(os.path.join(src_folder, label_filename))
            except IOError:
                mask_img = Image.new('RGB', tar_size)
            mask_img = mask_img.crop(rect)
            mask_img.save(os.path.join(save_folder, tmp_name+'-mask.png'))

            region = img.crop(rect)
            region.save(os.path.join(save_folder, tmp_name+'.png'))


def handle_image_v2(filename,src_folder,save_folder,tar_size=(64,64),stride=None,is_center=False):
    
    stride = tar_size[0]/2 if stride is None else stride

    # for 1-class segmentation case, 1) cut randomly, 2) cut with the center being a defect
    # 2021/10/26 if both img and mask are black, then do not generate
    img = Image.open(os.path.join(src_folder, filename))
    H, W = img.size 
    # if img.mode != 'RGB':
    #     img = img.convert('RGB')

    # 1) cut randomly
    for ii in range(ceil((H-tar_size[0])/stride)+1):
        for jj in range(ceil((W-tar_size[1])/stride)+1):
            start_h = ii*stride
            start_w = jj*stride
            end_h = ii*stride + tar_size[0]
            end_w = jj*stride + tar_size[1]
            if end_h > H:
                start_h = H - tar_size[0]
                end_h = H
            if end_w > W:
                start_w = W - tar_size[1]
                end_w = W
            rect = (start_w, start_h, end_w, end_h)
            tmp_name = filename[:11]+'_'+str(ii)+'_'+str(jj)

            region = img.crop(rect)
            if max(max(region.getextrema())) >0 :
                region.save(os.path.join(save_folder, tmp_name+'.png'))
                # save masks for patchs
                label_filename = filename[:11]+'-mask.png'
                mask_img = Image.open(os.path.join(src_folder, label_filename)).convert('L')
                mask_region = mask_img.crop(rect)
                mask_region.save(os.path.join(save_folder, tmp_name+'-mask.png'))

    # 2)cut with the center to be a label
    if is_center:
        centers = []
        mask_img = Image.open(os.path.join(src_folder, label_filename)).convert('L')
        mask_np = np.array(mask_img)
        lbl = label(mask_np)
        props = regionprops(lbl)

        for prop in props:
            centers.append(((prop.bbox[3]+prop.bbox[1])/2, (prop.bbox[2]+prop.bbox[0])/2))

        for nt, cc in enumerate(centers):
            tmp_name = filename[:11]+'_'+str(nt)
            left = int(cc[0]) - tar_size[0]/2
            right = int(cc[0]) + tar_size[0]/2
            top = int(cc[1]) -tar_size[1]/2
            bottom = int(cc[1]) + tar_size[0]/2
            rect = (left, top, right, bottom)

            region = img.crop(rect)
            region.save(os.path.join(save_folder,tmp_name+'.png'))

            mask_region = mask_img.crop(rect)
            mask_region.save(os.path.join(save_folder, tmp_name+'-mask.png'))

    # print(f'{filename[:11]} is_center={is_center} finish!')


def main():
    f = open(os.path.join(src_folder, "id.txt"))

    for line in f:
        filename = line[:-1]
        label_filename = filename[:11] + "-mask.png"
        # try:
        print(filename)
        handle_image(filename, label_filename, save_folder=save_folder)
        # except IOError:

    f.close()

if __name__ == "__main__":
    main()