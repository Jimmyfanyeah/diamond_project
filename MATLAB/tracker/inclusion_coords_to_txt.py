# -*- coding: utf-8 -*-

from PIL import Image
import os
import numpy as np
from skimage.segmentation import mark_boundaries as mkbdy
from skimage.measure import label, regionprops
import shutil


def inclusion_coords_to_txt(frame_path, mask_path, tmp_txt_save_path, vis_save_path):

    os.makedirs(vis_save_path, exist_ok=True)
    os.makedirs(tmp_txt_save_path, exist_ok=True)
    exist_list = os.listdir(tmp_txt_save_path)
    # exist_list = []
    mask_list = [n for n in os.listdir(mask_path) if not n[:11] in exist_list]
    
    for nn in mask_list:
        mask = Image.open(os.path.join(mask_path,nn))
        mask_np = np.array(mask)
        
        ############# visualization ##############
        img_path = os.path.join(frame_path,nn[:11]+'_001.png')
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        img_mask = mkbdy(img_np,mask_np[:,:,0],color=(1,0,0),outline_color=(1,0,0))
        img_mask = mkbdy(img_mask,mask_np[:,:,1],color=(0,1,0),outline_color=(0,1,0))
        img_mask = Image.fromarray((img_mask*255).astype('uint8'))
        
        img_mask.save(os.path.join(vis_save_path, nn[:11]+'_001_new_mask_vis.png'))
        
        ############ extract txt file ###############
        txt_save_path = os.path.join(tmp_txt_save_path,nn[:11])
        if os.path.isdir(txt_save_path):
            shutil.rmtree(txt_save_path)
        os.makedirs(txt_save_path)
        
        ## 0=R=reflection 1=G=inclusion
        # cnt = 0
        # for channel in range(2):
        #     mask_single = mask_np[:,:,channel]
        #     mask_single [mask_single>0] = 255
            
        #     boxes = []
        #     lbl = label(mask_single)
        #     props = regionprops(lbl)
            
        #     for prop in props:
        #         if prop.coords.shape[0]>1:
        #             save_name = os.path.join(txt_save_path, f'{nn[:11]}_{cnt}_{channel}.txt')
        #             np.savetxt(save_name, prop.coords, delimiter=' ', fmt='%d') 
        #             cnt += 1
        
        cnt = 0
        channel = 1
        mask_single = mask_np[:,:,channel]
        mask_single [mask_single>0] = 255
        
        lbl = label(mask_single)
        props = regionprops(lbl)
        
        for prop in props:
            if prop.coords.shape[0]>1:
                save_name = os.path.join(txt_save_path, f'{nn[:11]}_{cnt}_{channel}.txt')
                np.savetxt(save_name, prop.coords, delimiter=' ', fmt='%d') 
                cnt += 1
        
        channel = 0
        mask_single = mask_np[:,:,channel]
        mask_single [mask_single>0] = 255
        
        lbl = label(mask_single)
        props = regionprops(lbl)
        
        for prop in props:
            if prop.coords.shape[0]>1:
                save_name = os.path.join(txt_save_path, f'{nn[:11]}_{cnt}_{channel}.txt')
                np.savetxt(save_name, prop.coords, delimiter=' ', fmt='%d') 
                cnt += 1
        print(f'{nn}')

        
    
def read_label_to_box(src_folder, save_folder):

    os.makedirs(save_folder, exist_ok=True)
    
    # Run through all folders
    all_case = os.listdir(src_folder)
    exist_case = [n[:11] for n in os.listdir(save_folder)]
    all_case = list(set(all_case).difference(exist_case))
    
    for i in range(len(all_case)):
        case_folder = os.path.join(src_folder,all_case[i])
        
        save_case_folder = os.path.join(save_folder, all_case[i] + '_template')
        os.makedirs(save_case_folder, exist_ok=True)
        
        files = os.listdir(case_folder)
        for Files in files:
            print(Files)
            defect = np.genfromtxt(os.path.join(case_folder, Files), delimiter=' ')
            min_r = np.min(defect[:, 0])
            max_r = np.max(defect[:, 0])
            min_c = np.min(defect[:, 1])
            max_c = np.max(defect[:, 1])
            defect_img = np.zeros((int(max_r-min_r+1), int(max_c-min_c+1)))
            row = defect[:,0] - min_r
            col = defect[:,1] - min_c
            for j in range(len(row)):
                defect_img[int(row[j]), int(col[j])] = 255
            defect_img = np.pad(defect_img, 4, 'constant', constant_values=(0,0))
            defect_img = Image.fromarray(defect_img)
            if defect_img.mode != 'RGB':
                defect_img = defect_img.convert('RGB')
            filename = open(os.path.join(save_case_folder, Files), 'w')
            filename.write(
                str(int(min_c - 4)) + ' ' + str(int(min_r - 4)) + ' ' + str(int(max_c - min_c + 8)) + ' ' + str(
                    int(max_r - min_r + 8)) + '\n')
            filename.close()
            defect_img.save(os.path.join(save_case_folder, Files[:-4] + '.png'))



def main():
    
    frame_path = r'F:\Diamond\1stFrame'
    
    idx = 591
    mask_path = r'F:\Diamond\frame_mask\mask_new_'+str(idx)
    tmp_txt_save_path = r'F:\Diamond\frame_mask\txt_files_'+str(idx)
    
    vis_save_path = r'F:\Diamond\frame_mask\frame001_vis_'+str(idx)
    inclusion_coords_to_txt(frame_path, mask_path, tmp_txt_save_path, vis_save_path)
    
    src_folder = tmp_txt_save_path
    save_folder = os.path.join(r'F:\Diamond\trajectory_1213\_'+str(idx), 'inclusion_bbox')
    read_label_to_box(src_folder, save_folder)


if __name__ == "__main__":
	main()



