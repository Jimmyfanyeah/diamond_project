import os
from shutil import copy2,rmtree
from generate_mask_image import genMaskImage
from PIL import Image
import torchvision.transforms.functional as TTF
import torch
from generate_clipped_img import handle_image as clip_image
from clip_and_resize import handle_image

def gather(baseDir,saveDir,tarExtensions=['xml','png']):
    copyInfo = dict()
    for key in tarExtensions:
        copyInfo[key] = 0

    os.makedirs(saveDir,exist_ok=True)
    for root,_dirs,files in os.walk(baseDir):
        if 'test' not in root.lower():
            for file in files:
                if 'mask' not in file and file.split('.')[-1] in tarExtensions:
                    copy2(os.path.join(root,file),saveDir)
                    copyInfo[file.split('.')[-1]] += 1

    print('Copy finished!')
    print(copyInfo)

def checkXML(filelist):
    status = None
    for f in filelist:
        if f.split('.')[-1] == 'xml':
            status = f

    return status

def saveMask(root,files,saveDir):
    os.makedirs(saveDir,exist_ok=True)
    for f in files:
        if f.split('.')[-1] == 'png':
            if 'mask' in f:
                label = TTF.to_tensor(Image.open(os.path.join(root,f))).sum(dim=0,keepdim=True)
                label = label>0.0001
                label = label.float()
                label = torch.cat([label,]*3,dim=0)
                label = TTF.to_pil_image(label)
                label.save(os.path.join(saveDir,f[:11]+'-mask_v1.png'))
            else:
                copy2(os.path.join(root,f),os.path.join(saveDir,f[:11]+'.png'))


if __name__=='__main__':
    '''
    saveDir: diamond images and labels, with no crust and resized to 1200*1200
    clippedDir: diamond images and labels, clipped into 400*400
    testDir: diamond images and labels, with no crust and resized to 1200*1200, intended for testing network
    '''

    baseDir = '/media/hdd/css_data/hdr_data'
    labelDir = '/home/lingjia/Documents/CSS-PROJECT-DATA/UNET1/diamonds_labels'
    saveDir = '/home/lingjia/Documents/CSS-PROJECT-DATA/UNET1/diamonds_labels_cutted'
    clippedDir = '/home/lingjia/Documents/CSS-PROJECT-DATA/UNET1/diamonds_labels_clipped'
    testDir = '/home/lingjia/Documents/CSS-PROJECT-DATA/UNET1/blind_test'

    case = 'train' # test or train or frame.
    if case == 'train':
        if os.path.exists(labelDir):
            rmtree(labelDir)  # try to clean tmp folders

        for root,dirs,files in os.walk(baseDir):
            status = checkXML(files)
            if status is not None:
                genMaskImage(os.path.join(root,status),labelDir)
                for f in files:
                    if f.split('.')[-1] == 'png':
                        copy2(os.path.join(root,f),os.path.join(labelDir,f[:11]+'.png'))
            else:
                if 'test' not in root.lower():
                    saveMask(root,files,labelDir)

        # Delete crust (black borders) and resize diamonds to 1200*1200
        imgList = [f for f in os.listdir(labelDir) if 'mask' not in f]
        os.makedirs(saveDir,exist_ok=True)
        for f in imgList:
            handle_image(f,f[:11]+'-mask.png',labelDir,saveDir,tar_size=(1200,1200))

        # For training, we clip the images into 400*400 patches
        imgList = [f for f in os.listdir(saveDir) if 'mask' not in f and 'png' in f]
        os.makedirs(clippedDir,exist_ok=True)
        for f in imgList:
            clip_image(f,f[:11]+'-mask.png',saveDir,clippedDir)

    elif case == 'test':
        baseDir = '/home/lingjia/Documents/CSS_Project_1_1847/frame/1847test/1stframe'
        tmpSaveDir = '/home/lingjia/Documents/CSS_Project_1_1847/frame/1847test/1stframe_labels'
        testDir = '/home/lingjia/Documents/CSS_Project_1_1847/frame/1847test/1stframe_cutted'
        try:
            rmtree(tmpSaveDir)  # try to clean tmp folder
        except:
            pass

        # generate black 'fake masks'
        os.makedirs(tmpSaveDir,exist_ok=True)
        for root,dirs,files in os.walk(baseDir):
            saveMask(root,files,tmpSaveDir)

        # Delete crust (black borders) and resize diamonds to 1200*1200
        imgList = [f for f in os.listdir(tmpSaveDir) if 'mask' not in f]
        os.makedirs(testDir,exist_ok=True)
        for f in imgList:
            handle_image(f,f[:11]+'-mask.png',tmpSaveDir,testDir)

        image_dirs = os.listdir(testDir)
        # generate txt files
        with open(os.path.join(testDir,'id.txt'),'w') as file:
            for image_idx in image_dirs:
                if 'mask' not in image_idx and 'png' in image_idx:
                    file.write(image_idx)
                    file.write('\n')


    else:
        baseDir = '/home/lingjia/Documents/tmp/921_B/frame'
        testDir = '/home/lingjia/Documents/tmp/921_B/cutted'

        # Delete crust (black borders) and resize diamonds to 1200*1200
        imgList = [f for f in os.listdir(baseDir) if 'png' in f]
        os.makedirs(testDir,exist_ok=True)
        for f in imgList:
            handle_image_v2(f,f.split('.')[0]+'-mask.png',baseDir,testDir)




