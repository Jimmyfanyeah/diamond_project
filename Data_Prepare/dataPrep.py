import os
from shutil import copy2,rmtree
from gen_mask_image import genMaskImage
from cut_and_resize import handle_image
from clip_img import handle_image_v2 as clip_image
import argparse

def checkXML(filelist):
    status = None
    for f in filelist:
        if f.split('.')[-1] == 'xml':
            status = f

    return status


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Diamond Project')
    parser.add_argument('--phase',          type=str,           default='other',        help='train or test')
    opt = parser.parse_args()

    # For train
    '''
    baseDir: data path for original diamond images and videos
    saveDir: copy diamond images from baseDir and generate corresponding image
    clippedDir: diamond images and labels, clipped into 400*400
    testDir: diamond images and labels, with no crust and resized to 1200*1200, intended for testing network
    '''
    baseDir = '/media/hdd_4T/css_data/hdr_data_v2'
    # baseDir = '/media/hdd_4T/css_data/hdr_data_v2/20191204 40 HDR images and videos (1 - 40)/20191023 (20p)'
    labelDir = '/media/hdd_4T/css_data/1class/initial'
    cutDir = '/media/hdd_4T/css_data/1class/cut'
    clipDir = '/media/hdd_4T/css_data/1class/clip128'

    if opt.phase == 'train':
        if os.path.exists(labelDir):
            print('label exist!')
            exit()

        # Generate masks
        print('STEP 1 Generate masks!')
        for root,dirs,files in os.walk(baseDir):
            status = checkXML(files)
            if status is not None:
                genMaskImage(os.path.join(root,status),labelDir)
                for f in files:
                    if f.split('.')[-1] == 'png' and 'mask' not in f:
                        copy2(os.path.join(root,f),os.path.join(labelDir,f[:11]+'.png'))

        # Delete crust (black borders) and resize diamonds to 1024*1024 (if applicable)
        print('STEP 2 Delete crust!')
        imgList = [f for f in os.listdir(labelDir) if 'mask' not in f]
        os.makedirs(cutDir,exist_ok=True)
        for f in imgList:
            handle_image(f,f[:11]+'-mask.png',labelDir,cutDir,is_resize=False, tar_size=(1024,1024))

        # For training, we clip the images into small size patchs
        imgList = [f for f in os.listdir(cutDir) if 'mask' not in f and 'png' in f]
        os.makedirs(clipDir,exist_ok=True)
        for idx, f in enumerate(imgList):
            clip_image(f,cutDir,clipDir,tar_size=(128,128),stride=64,is_center=True)
            print(f'{idx}/{len(imgList)-1} {f[:11]} finish!')


    elif opt.phase == 'test':
        # For test/infer
        baseDir = '/media/hdd/css/frame001/white'
        cutDir = '/media/hdd/css/frame001/white_cut'

        # Delete crust (black borders) and resize diamonds to 1024*1024 (optional)
        imgList = [f for f in os.listdir(baseDir) if 'png' in f]
        os.makedirs(cutDir,exist_ok=True)
        for idx, f in enumerate(imgList):
            handle_image(f,f.split('.')[0]+'-mask.png',baseDir,cutDir,is_resize=False, tar_size=(1024,1024))

        # Generate txt files
        imgList = os.listdir(cutDir)
        imgList.sort()
        with open(os.path.join(cutDir,'id.txt'),'w') as file:
            for f in imgList:
                if 'mask' not in f and 'png' in f:
                    file.write(f+'\n')

    else:
        print(f'No such process as {opt.phase}!')




