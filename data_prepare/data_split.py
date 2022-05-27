import os
import random
from shutil import move,copy2
from math import ceil

def save_list(imgList,txt_name):
    with open(txt_name,'w') as file:
        for term in imgList:
            file.write(term+'\n')
    print(f'save {txt_name}')


def split_train_val_set(base_path,fileList=None,fold=10,val_fold=2,test_fold=4,seed=10):
    fileList = os.listdir(base_path) if fileList is None else fileList

    num = len(fileList)
    step = ceil(num/fold)

    fileList.sort()
    random.seed(seed)
    random.shuffle(fileList)
    valList = fileList[(val_fold-1)*step :min(val_fold*step, num)]
    testList = fileList[(test_fold-1)*step :min(test_fold*step, num)]
    trainList = list(set(fileList).difference(valList).difference(testList))

    save_list(trainList, os.path.join(base_path,'train_ids.txt'))
    save_list(valList, os.path.join(base_path,'val_ids.txt'))
    save_list(testList, os.path.join(base_path,'test_ids.txt'))

    print('Num of all:{} train:{}, val:{}, test:{}'.format(num,len(trainList),len(valList),len(testList)))


if __name__ == '__main__':
    
    base_path = '/media/hdd/diamond_data/cls_multi-class_EfficientNet'

    """ Split it to train & val for each type of inclusion """
    # folderList = [n for n in os.listdir(base_path) if not 'txt' in n]
    # for folder in folderList:
    #     tmp_path = os.path.join(base_path,folder)
    #     fileList = os.listdir(tmp_path)
    #     split_train_val_set(tmp_path, fileList)

    """ Merge types into train & val groups """
    # oao_groups = {'1':['Cloud','Crystal'], '2':['Cloud','Twinning_wisp'], '3':['Cloud','Feather'], 
    #         '4':['Crystal','Twinning_wisp'], '5':['Crystal','Feather'], '6':['Feather','Twinning_wisp']}
    # phases = ['train','val','test']

    # base_path = base_path
    # save_path = '/media/hdd/diamond_data/cls_multi-class_EfficientNet_OAO_strategy_groups'
    # for gidx in oao_groups.keys():
    #     oaog = oao_groups[gidx]
    #     for phase in phases:
    #         for cls_name in oaog:
    #             tmp_src_path = os.path.join(base_path,cls_name)
    #             tmp_save_path = os.path.join(save_path,f'{oaog[0]}_{oaog[1]}',phase,cls_name)
    #             os.makedirs(tmp_save_path, exist_ok=True)

    #             with open(os.path.join(tmp_src_path,f'{phase}_ids.txt'),'r') as file:
    #                 tmpList = file.readlines()
    #             tmpList = [n.split('\n')[0] for n in tmpList]

    #             for file in tmpList:
    #                 copy2(os.path.join(tmp_src_path,file),os.path.join(tmp_save_path,file))
    #     print(f'Finish {oaog}')

    """ Multi-class"""
    # class_list = ['Pinpoint','Crystal','Needle','Feather','Internal_graining',
    #             'Cloud','Twinning_wisp','Nick','Pit','Burn_mark']
    class_list = ['Crystal','Feather','Cloud','Twinning_wisp']
    phases = ['train','val','test']

    base_path = base_path
    save_path = '/media/hdd/diamond_data/cls_multi-class_EfficientNet_OAO_strategy_groups/multiclass_once'
    for phase in phases:
        for cls_name in class_list:
            tmp_src_path = os.path.join(base_path,cls_name)
            tmp_save_path = os.path.join(save_path,phase,cls_name)
            os.makedirs(tmp_save_path, exist_ok=True)
            with open(os.path.join(tmp_src_path,f'{phase}_ids.txt'),'r') as file:
                tmpList = file.readlines()
            tmpList = [n.split('\n')[0] for n in tmpList]

            for file in tmpList:
                copy2(os.path.join(tmp_src_path,file),os.path.join(tmp_save_path,file))
    print(f'Finish multiclass setting')
