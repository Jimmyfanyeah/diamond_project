# sensitivity for diamond project
from PIL import Image
import numpy as np
import skimage.measure as measure
import os

def singleImgCheck(pred_path,gt_path):
    TP = 0
    # pred_path = r'C:\Users\mastaffs\Desktop\10364964462-pred.png'
    # gt_path = r'C:\Users\mastaffs\Desktop\10364964462-mask.png'
    pred = Image.open(pred_path)
    pred_np = np.array(pred)
    label_pred = measure.label(pred_np)
    props_pred = measure.regionprops(label_pred)

    mask = Image.open(gt_path).convert('L')
    mask_np = np.array(mask)
    label_mask = measure.label(mask_np)
    props = measure.regionprops(label_mask)
    for jj in range(len(props)):
        gt_bbox = props[jj].bbox
        # num of pixel for the region, bbox_area = num of pixel for bbox
        gt_area = props[jj].area  
        # print('gt {} {}'.format(gt_bbox,gt_area))

        for ii in range(len(props_pred)):
            pred_bbox = props_pred[ii].bbox
            pred_area = props_pred[ii].area

            dist_check = [pred_bbox[i]-gt_bbox[i]<10 for i in range(2)]+ [pred_bbox[i]-gt_bbox[i]>-10 for i in range(2,4)]
            area_check = abs(pred_area-gt_area)<max(300,gt_area*0.5)

            if min(dist_check) and area_check:
                TP = TP + 1
                # print('pred {} {}'.format(pred_bbox,pred_area))
                break
    # print('all = {}, tp = {}'.format(ALL, TP))
    ALL = len(props)
    return ALL, TP

pred_path = '/home/lingjia/Documents/chow/unet1_results/Images/0403_0127BestDiceAll'
gt_path = '/home/lingjia/Documents/Diamond_Project_Data/UNET1/diamonds_labels_cutted'
save_file_path = '/home/lingjia/Documents/chow/unet1_results/Images/0403_0127BestDiceAll.txt'
num_all = 0
num_tp = 0
num_fn = 0

img_list = list(set([int(n[:11]) for n in os.listdir(pred_path) if 'png' in n]))
img_list.sort()
# img_list = ['10345832292']
file = open(save_file_path,'w')
for idx, img_idx in enumerate(img_list):
    img_idx = str(img_idx)
    pred_img = os.path.join(pred_path,img_idx+'-pred.png')
    gt_img = os.path.join(gt_path,img_idx+'-mask.png')

    all_tmp, tp_tmp = singleImgCheck(pred_img, gt_img)
    num_all = num_all + all_tmp
    num_tp = num_tp + tp_tmp
    num_fn = num_fn + all_tmp - tp_tmp

    sens = 100
    if all_tmp >0 :
        sens = tp_tmp/all_tmp
    file.write('{} sens={}, nall={}, tp={}\n'.format(img_idx, sens, all_tmp, tp_tmp))
    print('{} sens={}, num_all={}, num_tp={}'.format(img_idx, sens, all_tmp, tp_tmp))

    # if idx%50 == 0:
    #     sens = num_tp/num_all
    #     print('{} sens={}, num_all={}, num_tp={}'.format(idx, sens, num_all, num_tp))
sens = num_tp/num_all
print('{} sens={}, num_all={}, num_tp={}'.format(idx, sens, num_all, num_tp))
file.write('{} sens={}, num_all={}, num_tp={}\n'.format(idx, sens, num_all, num_tp))
file.close()