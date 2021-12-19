

# 1 = generate id.txt for 1 folder randomly
# 2 = generate id.txt, id_val.txt, id_test.txt for cutted and clipped folder
case = '2'

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
    test_fold = 5

    base_folder = '/home/lingjia/Documents/diamond_data/1class'
    cut_folder = os.path.join(base_folder,'cut')
    clip_folder = os.path.join(base_folder,'clip')

    special_case = []
    # special_cases = ['10353219437','10371944258']
    test_special = []
    # with open('/home/lingjia/Documents/chow/unet/Data_Prepare/ids_fp.txt','r') as file:
    #     for line in file:
    #         special_case.append(line[:11])
    # with open('/home/lingjia/Documents/diamond/unet/Data_Prepare/ids_large_regions.txt','r') as file:
    #     for line in file:
    #         special_case.append(line[:11])
    # with open('/home/lingjia/Documents/chow/unet/Data_Prepare/ids_twinning_wisp.txt','r') as file:
    #     for line in file:
    #         special_case.append(line[:11])
    # with open('/home/lingjia/Documents/chow/unet/Data_Prepare/special_cases_IG.txt','r') as file:
    #     for line in file:
    #         special_case.append(line[:11])

    print('num of special case: {}'.format(len(special_case)))

    imgList = list(set([name[:11] for name in os.listdir(cut_folder) if not 'mask' in name and 'png' in name]))
    n_case = len(imgList)
    step = ceil(n_case/n_fold) #ceil(4.1)=5

    all_case = list(set(imgList).difference(set(special_case)))
    all_case.sort()
    random.seed(seed)
    random.shuffle(all_case)

    val_case = all_case[(val_fold-1)*step :min(val_fold*step, n_case)]
    test_case = all_case[(test_fold-1)*step :min(test_fold*step, n_case)]
    print('num of all:{}, val: {}, test: {}'.format(n_case-len(val_case)-len(test_case), len(val_case), len(test_case)))

    # generate txt files for cutted folder
    imgList = [name for name in os.listdir(cut_folder) if not 'mask' in name and 'png' in name]
    id_train_file = open(os.path.join(cut_folder,'id_train.txt'),'w')
    id_val_file = open(os.path.join(cut_folder,'id_val.txt'),'w')
    id_test_file = open(os.path.join(cut_folder,'id_test.txt'),'w')
    for image_idx in imgList:
        if image_idx[:11] in test_case or image_idx[:11] in test_special:
            id_test_file.write(image_idx)
            id_test_file.write('\n')
        elif image_idx[:11] in val_case:
            id_val_file.write(image_idx)
            id_val_file.write('\n')
        else:
            id_train_file.write(image_idx)
            id_train_file.write('\n')

    id_train_file.close()
    id_val_file.close()
    id_test_file.close()

    ## generate txt files for clip folder
    sum_train, sum_val, sum_test = 0,0,0
    imgList = [name for name in os.listdir(clip_folder) if not 'mask' in name and 'png' in name]
    id_train_file = open(os.path.join(clip_folder,'id_train.txt'),'w')
    id_val_file = open(os.path.join(clip_folder,'id_val.txt'),'w')
    id_test_file = open(os.path.join(clip_folder,'id_test.txt'),'w')

    for image_idx in imgList:
        if image_idx[:11] in test_case or image_idx[:11] in test_special:
            id_test_file.write(image_idx)
            id_test_file.write('\n')
            sum_test = sum_test+1
        elif image_idx[:11] in val_case:
            id_val_file.write(image_idx)
            id_val_file.write('\n')
            sum_val = sum_val+1
        else:
            id_train_file.write(image_idx)
            id_train_file.write('\n')
            sum_train = sum_train+1

    id_train_file.close()
    id_val_file.close()
    id_test_file.close()

    print('for small patch, num of train:{}, val:{}, test:{}'.format(sum_train,sum_val,sum_test))
