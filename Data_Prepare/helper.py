import os
from PIL import Image
import random
from math import ceil
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as ttf


"""data preprocessing"""
def count_mask_with_target(data_path,fileids=None):
    if fileids is None:
        fileids = [n for n in os.listdir(data_path) if 'mask' in n and 'png' in n]

    # compute # of mask with label / without label
    with_target = 0
    ii = 0
    for file in fileids:
        img = Image.open(os.path.join(data_path,file)).convert('L')
        if img.getextrema()[1] > 0:
            with_target = with_target + 1
        ii = ii+1
        if ii%1e5 ==0:
            print(f'number = {with_target}:{ii} - ratio = {with_target/ii:.3f}:{1-with_target/ii:.3f}')
    return with_target


def split_imgList(data_path,imgList=None,baseDict=None,seed=10,nFold=10,valFold=1,testFold=5):
    # split = generate id.txt, id_val.txt, id_test.txt for cut and clip folder
    if imgList is None:
        imgList = list(set([n[:11] for n in os.listdir(data_path) if not 'mask' in n and 'png' in n]))

    if baseDict is not None:
        print('Based on base dict')
        base_trainList = baseDict['train']
        base_valList = baseDict['val']
        base_testList = baseDict['test']

        trainList = [n for n in imgList if n[:11] in base_trainList]
        valList = [n for n in imgList if n[:11] in base_valList]
        testList = [n for n in imgList if n[:11] in base_testList]
        print(f'data path: {data_path},\nNum of train:{len(trainList)}, val:{len(valList)}, test:{len(testList)}')
        print(f'some example:{trainList[1:3]}')

    else:
        print('From scratch!')
        nCase = len(imgList)
        step = ceil(nCase/nFold)
        # imgList = list(set(imgList).difference(set(special_case)))
        imgList.sort()
        seed=10
        random.seed(seed)
        random.shuffle(imgList)

        valList = imgList[(valFold-1)*step :min(valFold*step, nCase)]
        testList = imgList[(testFold-1)*step :min(testFold*step, nCase)]
        trainList = list(set(imgList).difference(set(valList)).difference(set(testList)))
        print(f'data path: {data_path},\nNum of train:{len(trainList)}, val:{len(valList)}, test:{len(testList)}')
        print(f'some example:{trainList[1:3]}')

    return trainList, valList, testList


def save_list(imgList,txt_name):
    with open(txt_name,'w') as file:
        for term in imgList:
            file.write(term+'\n')
    print(f'save {txt_name}')


def adjust_ratio(data_path,txt_save_path,ratio=1):
    # Since in phase (train/val) set, number of samples with target & without target is not equal -> heavily unbalanced dataset
    # Given ratio, pick samples follow with : without target = ratio
    for phase in ['train','val']:
        with_target, without_target = 0, 1
        id_file = open(os.path.join(txt_save_path,'id_'+phase+'_clip.txt'),'r')
        id_adjust_file = open(os.path.join(txt_save_path,'id_'+phase+'_clip_adjust.txt'),'w')
        for line in id_file:
            mask_name = line.split('.')[0]+'-mask.png'
            img = Image.open(os.path.join(data_path,mask_name)).convert('L')
            if img.getextrema()[1] > 0:
                with_target = with_target + 1
                id_adjust_file.write(line)
            elif with_target/without_target >= ratio:
                without_target = without_target + 1	
                id_adjust_file.write(line)

        id_file.close()
        id_adjust_file.close()
        print(f'{phase}: with inclusion = {with_target}, without = {without_target}')


class MyDataset(Dataset):
    def __init__(self, data_dir, data_id_list):
        self.img_dict, self.mask_dict = {},{}
        self.data_id_list = data_id_list
        self.data_dir = data_dir
        for case in data_id_list:
            self.img_dict[case] = case

    def __getitem__(self,idx):
        data_idx = self.data_id_list[idx]
        img = Image.open(os.path.join(self.data_dir, self.img_dict[data_idx])).resize((512, 512),Image.LANCZOS)
        img = ttf.to_tensor(img)
        return img.float()

    def __len__(self):
        return len(self.data_id_list)


def check_mean_std(data_path,id_file=None):
    if id_file is not None:
        with open(id_file,'r') as file:
            data_id_list = file.read().split('\n')
        print(f'Check mean and std for {id_file}')
    else:
        data_id_list = [n for n in os.listdir(data_path) if not 'mask' in n and 'png' in n]
        print(f'Check mean and std for all imgs in {data_path}')

    print(f'Length of data_id_list = {len(data_id_list)}')
    dataset = MyDataset(data_path, data_id_list)
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(f'mean: {mean}')
    print(f'std: {std}')


if __name__=='__main__':
    # Preprocess
    # data_path = '/media/hdd/css_data/1class'
    # data_path = '/media/hdd_4T/css_data/1class'
    cut_path = os.path.join(data_path,'cut')
    clip_path = os.path.join(data_path,'clip128')
    txt_save_path = os.path.join(data_path,'txt128')
    os.makedirs(txt_save_path, exist_ok=True)

    # """Step 1 - Count information"""
    # print('STEP 1 - Count information')
    # # Cound number of samples
    # imgList = [n for n in os.listdir(cut_path) if not 'mask' in n and 'png' in n]
    # print(f'cut folder - {cut_path} \nSamples = {len(imgList)}')
    # imgList_random = [n for n in os.listdir(clip_path) if not 'mask' in n and 'png' in n and len(n.split('_'))==3]
    # print(f'clip folder - {clip_path} \nRandom patches = {len(imgList_random)}')
    # imgList_center = [n for n in os.listdir(clip_path) if not 'mask' in n and 'png' in n and len(n.split('_'))==2]
    # print(f'Centered patches = {len(imgList_center)}')

    # # Count number of samples with inclusion
    # num_with_target = count_mask_with_target(cut_path)
    # print(f'cut folder - {cut_path} \nWith target samples = {num_with_target}')
    # imgList_random = [n for n in os.listdir(clip_path) if 'mask' in n and 'png' in n and len(n.split('_'))==3]
    # num_with_target = count_mask_with_target(clip_path,imgList_random)
    # print(f'clip folder - {clip_path} \nWith target samples random = {num_with_target}')
    # imgList_center = [n for n in os.listdir(clip_path) if 'mask' in n and 'png' in n and len(n.split('_'))==2]
    # num_with_target = count_mask_with_target(clip_path,imgList_center)
    # print(f'With target samples center = {num_with_target}')


    # """Step 2 - Split IDs to train, val, test set"""
    # print('STEP 2 - Split IDs to train, val and test set, save txt file')
    # # Generate train, val and test set for cut folder
    # # imgList = [n for n in os.listdir(cut_path) if not 'mask' in n and 'png' in n]
    # trainList, valList, testList = split_imgList(cut_path)
    # save_list(trainList,os.path.join(txt_save_path,'id_train_cut.txt'))
    # save_list(valList,os.path.join(txt_save_path,'id_val_cut.txt'))
    # save_list(testList,os.path.join(txt_save_path,'id_test_cut.txt'))

    # imgList_random = [n for n in os.listdir(clip_path) if not 'mask' in n and 'png' in n and len(n.split('_'))==3]
    # baseDict = {'train':trainList, 'val':valList, 'test':testList}
    # clip_trainList, clip_valList, clip_testList = split_imgList(clip_path,imgList_random,baseDict)
    # save_list(clip_trainList,os.path.join(txt_save_path,'id_train_clip.txt'))
    # save_list(clip_valList,os.path.join(txt_save_path,'id_val_clip.txt'))
    # save_list(clip_testList,os.path.join(txt_save_path,'id_test_clip.txt'))


    """Step 3 - Given ratio = with : without target, pick out examples as training set"""
    print('---> STEP 3 - Adjust to ratio')
    ratio = 2
    adjust_ratio(clip_path,txt_save_path,ratio=ratio)


    """Step 4 Check mean and std for train dataset"""
    # print('STEP 4 - Check mean and std')
    # id_file = os.path.join(txt_save_path,'id_val_clip.txt')
    # check_mean_std(clip_path,id_file)