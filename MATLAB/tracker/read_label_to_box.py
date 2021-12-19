import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# Folder location
src_folder = r'F:\Diamond\frame_mask\txt_files5'
save_folder = r'F:\Diamond\trajectory_1213\inclusion_bbox'

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


