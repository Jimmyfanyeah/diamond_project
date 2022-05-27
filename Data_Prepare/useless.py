############# others


def copy_files(base_path,save_path,txt_file=None):
    # 20210405 copy videos with IG to folder
    try:
        rmtree(save_path)  # try to clean tmp folders
    except:
        pass
    os.makedirs(save_path,exist_ok=True)

    cases = []
    if txt_file:
        with open(txt_file,'r') as file:
            for line in file:
                cases.append(line[:-1])

    for root,dirs,files in os.walk(base_path):
        for file in files:
            if 'mp4' in file and file[:11] in cases:
                copy2(os.path.join(root,file),os.path.join(save_path,file[:11]+'.mp4'))
                print(file)


def data_back_to_folder(data_path,mask_path,save_path):
    # put all mask image back to corresponding folder
    mask_list = [n[:11] for n in os.listdir(mask_path)]
    tmp_save_dir = save_path

    for root,dirs,files in os.walk(data_path):
        for file in files:
            if 'png' in file and file[:11] in mask_list:
                tmp_dirs = root.split('/')[5:]
                for tmp in tmp_dirs:
                    tmp_save_dir = os.path.join(tmp_save_dir,tmp)
                # print(tmp_save_dir+'\n')
                # overlap
                tmp_overlap_dir = os.path.join(tmp_save_dir,'overlap')
                os.makedirs(tmp_overlap_dir,exist_ok=True)
                copy2(os.path.join(mask_path,'overlap',file[:11]+'_double.png'),os.path.join(tmp_overlap_dir,file[:11]+'_pred.png'))
                # double
                tmp_double_dir = os.path.join(tmp_save_dir,'double')
                os.makedirs(tmp_double_dir,exist_ok=True)
                copy2(os.path.join(mask_path,'double',file[:11]+'_double.png'),os.path.join(tmp_double_dir,file[:11]+'_pred.png'))
            elif 'png' in file:
                print(os.path.join(root,file[:11]))
            tmp_save_dir = save_path


def pixel_acc_per_class():


    inclusion_type = 'Crystal'
    mask_dir = '/home/lingjia/Documents/tmp_2class/hdr_diamonds_labels_cutted/'+inclusion_type
    pred_dir = '/home/lingjia/Documents/chow/unet1_results/Images/0127BestDice'
    save_dir = '/home/lingjia/Documents/chow/unet1_results/Images/0127BestDice'
    file = open(os.path.join(save_dir,inclusion_type+'_acc.txt'),'w')
    file_detail = open(os.path.join(save_dir,inclusion_type+'_acc_detail.txt'),'w')

    mask_list = [n[:11] for n in os.listdir(mask_dir) if 'mask' in n]

    train_list = []
    val_list = []
    # train_list = [n[:11] for n in os.listdir(os.path.join(pred_dir,'train')) if 'png' in n]
    # val_list = [n[:11] for n in os.listdir(os.path.join(pred_dir,'val')) if 'png' in n]
    test_list = [n[:11] for n in os.listdir(os.path.join(pred_dir,'test')) if 'png' in n]
    train_list = [n for n in mask_list if n in train_list]
    val_list = [n for n in mask_list if n in val_list]
    test_list = [n for n in mask_list if n in test_list]

    print('{} img numbers mask:{} train:{} val:{} test:{}'.format(inclusion_type,len(mask_list),len(train_list),len(val_list),len(test_list)))
    file.write('{} mask:{} train:{} val:{} test:{}'.format(inclusion_type,len(mask_list),len(train_list),len(val_list),len(test_list)))
    file.write('\n')

    # compute accuracy for each set
    dict_list = {'val':val_list,'test':test_list,'train':train_list}
    for p_id,phase in enumerate(dict_list.keys()):
        label_list = []
        pred_list = []
        tmp_list = dict_list[phase]
        if len(tmp_list) == 0:
            continue
        print(phase)
        image_id = 0
        for idx in tmp_list:
            num_label = 0
            num_pred = 0

            mask = Image.open(os.path.join(mask_dir,idx+'-mask.png'))
            pred = Image.open(os.path.join(pred_dir,phase,idx+'-pred.png'))

            height, width = mask.size
            pred = pred.resize((height,width))

            mask = np.array(mask)
            pred = np.array(pred)

            # compute accuracy for defect
            for ii in range(height):
                for jj in range(width):
                    if mask[ii,jj,0]>0 or mask[ii,jj,1]>0:
                        num_label = num_label+1
                        if pred[ii,jj]>0:
                            num_pred = num_pred+1
            label_list.append(num_label)
            pred_list.append(num_pred)
            image_id = image_id+1
            if num_label>0:
                acc = num_pred/num_label
            else: acc = 2
            file_detail.write('{} {} {} {} {:.4f}'.format(p_id,idx,num_label,num_pred,acc))
            file_detail.write('\n')
            if image_id%10==0:
                print(image_id,num_label,num_pred,acc)
        if sum(label_list)>0:
            de_per = sum(pred_list)/sum(label_list)
        else: de_per = 2
        print('{} label_sum:{} intersect_sum:{} per:{:.4f}'.format(phase,sum(label_list),sum(pred_list),de_per))
        file.write('{} label_sum:{} intersect_sum:{} per:{:.4f}'.format(phase,sum(label_list),sum(pred_list),de_per))
        file.write('\n')

    file.close()
    file_detail.close()


    # # compute accuracy for train set
    # dict_list = {'val':val_list,'test':test_list,'train':train_list}
    # for p_id,phase in enumerate(dict_list.keys()):
    #     de_label_list = []
    #     de_pred_list = []
    #     re_label_list = []
    #     re_pred_list = []
    #     tmp_list = dict_list[phase]
    #     if len(tmp_list) == 0:
    #         continue
    #     print(phase)
    #     image_id = 0
    #     for idx in tmp_list:
    #         de_label = 0
    #         de_pred = 0
    #         re_label = 0
    #         re_pred = 0

    #         mask = Image.open(os.path.join(mask_dir,idx+'-mask.png'))
    #         pred = Image.open(os.path.join(pred_dir,phase,idx+'-pred.png'))

    #         height, width = mask.size
    #         pred = pred.resize((height,width))

    #         mask = np.array(mask)
    #         pred = np.array(pred)

    #         # compute accuracy for defect
    #         for ii in range(height):
    #             for jj in range(width):
    #                 if mask[ii,jj,0]>0:
    #                     de_label = de_label+1
    #                     if pred[ii,jj]>0:
    #                         de_pred = de_pred+1
    #                 elif mask[ii,jj,1]>0:
    #                     re_label = re_label+1
    #                     if pred[ii,jj]>0:
    #                         re_pred = re_pred+1
    #         de_label_list.append(de_label)
    #         de_pred_list.append(de_pred)
    #         re_label_list.append(re_label)
    #         re_pred_list.append(re_pred)
    #         image_id = image_id+1

    #         if de_label>0:
    #             acc_de = de_pred/de_label
    #         else: acc_de = 2
    #         if re_label>0:
    #             acc_re = re_pred/re_label
    #         else: acc_re = 2
    #         file_detail.write('{} {} {} {} {} {} {:.4f} {:.4f}'.format(p_id,idx,de_label,de_pred,re_label,re_pred,acc_de,acc_re))
    #         file_detail.write('\n')
    #         if image_id%10==0:
    #             print(image_id,de_label,de_pred,re_label,re_pred)
    #     if sum(de_label_list)>0:
    #         de_per = sum(de_pred_list)/sum(de_label_list)
    #     else: de_per = 2
    #     print('{} defect label_sum:{} intersect_sum:{} per:{:.4f}'.format(phase,sum(de_label_list),sum(de_pred_list),de_per))
    #     file.write('{} defect label_sum:{} intersect_sum:{} per:{:.4f}'.format(phase,sum(de_label_list),sum(de_pred_list),de_per))
    #     file.write('\n')
    #     if sum(re_label_list)>0:
    #         re_per = sum(re_pred_list)/sum(re_label_list)
    #     else: re_per = 2
    #     print('{} reflection label_sum:{} in_sum:{} per:{:.4f}'.format(phase,sum(re_label_list),sum(re_pred_list),re_per))
    #     file.write('{} reflection label_sum:{} in_sum:{} per:{:.4f}'.format(phase,sum(re_label_list),sum(re_pred_list),re_per))
    #     file.write('\n')

    # file.close()
    # file_detail.close()