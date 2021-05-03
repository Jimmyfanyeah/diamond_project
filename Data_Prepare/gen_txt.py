import os
import random
from math import ceil

# 1 = generate id.txt for 1 folder randomly
# 2 = generate id.txt, id_val.txt, id_test.txt for cutted and clipped folder
case = '1'
if case == '1':
    src_folder = '/home/lingjia/Documents/CSS-PROJECT-DATA/20210405_Needle/diamonds_labels'
    save_folder = '/home/lingjia/Documents/CSS-PROJECT-DATA/20210405_Needle'
    image_dirs = os.listdir(src_folder)

    # generate txt files
    with open(os.path.join(save_folder,'id.txt'),'w') as file:
        for image_idx in image_dirs:
            if 'mask' not in image_idx and 'png' in image_idx:
                file.write(image_idx[:11])
                file.write('\n')

elif case == '2':
    # all patches of 1 image in same phase, train or validation
    # save txt file for both cutted and clipped folder

    seed = 10
    n_fold = 10
    val_fold = 1
    test_fold = 3

    base_folder = '/home/lingjia/Documents/Diamond_Project_Data/UNET1'
    cut_folder = os.path.join(base_folder,'diamonds_labels_cutted')
    clip_folder = os.path.join(base_folder,'diamonds_labels_clipped')

    # special_cases = ['10353219437','10371944258']
    special_cases = []

    allcase = []
    image_dirs = [name for name in os.listdir(cut_folder) if not 'mask' in name and 'png' in name]
    for idx in image_dirs:
        if idx[:11] in special_cases:
            print('special '+idx[:11])
        elif idx[:11] in allcase:
            print('existed '+idx[:11])
        else:
            allcase.append(idx[:11])

    n_case = len(allcase)
    allcase.sort()
    random.seed(seed)
    random.shuffle(allcase)
    step = ceil(n_case/n_fold) #ceil(4.1)=5
    val_case = allcase[(val_fold-1)*step :min(val_fold*step, n_case)]
    test_case = allcase[(test_fold-1)*step :min(test_fold*step, n_case)]
    print('num of all:{}, validation: {}, test: {}'.format(n_case, len(val_case), len(test_case)))

    val_special = []
    test_special = special_cases

    # generate txt files for cutted folder
    val_sum = 0
    all_sum = 0
    test_sum = 0
    image_dirs = [name for name in os.listdir(cut_folder) if not 'mask' in name and 'png' in name]
    with_labels_file = open(os.path.join(cut_folder,'id_all.txt'),'w')
    id_validation_file = open(os.path.join(cut_folder,'id_val.txt'),'w')
    id_test_file = open(os.path.join(cut_folder,'id_test.txt'),'w')
    for image_idx in image_dirs:
        if image_idx[:11] in test_case or image_idx[:11] in test_special:
            id_test_file.write(image_idx)
            id_test_file.write('\n')
            test_sum = test_sum + 1
        else:
            with_labels_file.write(image_idx)
            with_labels_file.write('\n')
            all_sum = all_sum + 1
            if image_idx[:11] in val_case or image_idx[:11] in val_special:
                id_validation_file.write(image_idx)
                id_validation_file.write('\n')
                val_sum = val_sum + 1

    with_labels_file.close()
    id_validation_file.close()
    id_test_file.close()

    ## generate txt files for clip folder
    image_dirs = [name for name in os.listdir(clip_folder) if not 'mask' in name and 'png' in name]
    with_labels_file = open(os.path.join(clip_folder,'id_all.txt'),'w')
    id_validation_file = open(os.path.join(clip_folder,'id_val.txt'),'w')
    id_test_file = open(os.path.join(clip_folder,'id_test.txt'),'w')

    test_sum = 0
    val_sum = 0
    all_sum = 0
    for image_idx in image_dirs:
        if image_idx[:11] in test_case or image_idx[:11] in test_special:
            id_test_file.write(image_idx)
            id_test_file.write('\n')
            test_sum = test_sum+1
        else:
            with_labels_file.write(image_idx)
            with_labels_file.write('\n')
            all_sum = all_sum+1
            if image_idx[:11] in val_case or image_idx[:11] in val_special:
                id_validation_file.write(image_idx)
                id_validation_file.write('\n')
                val_sum = val_sum + 1

    with_labels_file.close()
    id_validation_file.close()
    id_test_file.close()

    print('clip folder, num of all:{}, val:{}, test:{}'.format(all_sum,val_sum,test_sum))
