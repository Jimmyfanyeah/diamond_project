# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:19:39 2021
@author: mastaffs
"""
import xml.dom.minidom
import numpy as np
import os
from skimage.measure import label, regionprops
from PIL import Image, ImageDraw

def checkXML(filelist):
    status = None
    for f in filelist:
        if f.split('.')[-1] == 'xml':
            status = f
    return status


def CutAndLabel(xmlpath,src_path,save_path,txt_save_path,resize=True,tar_size=(32,32),clear_dist=2):
    dom = xml.dom.minidom.parse(os.path.join(src_path,xmlpath))
    xml_root = dom.documentElement
    images = xml_root.getElementsByTagName('image')

    # Note: pinpoint is not considered here # 'Pinpoint':1,
    class_list = {'Pinpoint': 1, 'Crystal': 2, 'Needle': 3, 'Feather': 4, 'Internal_graining': 5,
                           'Cloud': 6, 'Twinning_wisp': 7, 'Nick': 8, 'Pit': 9, 'Burn_mark': 10}

    for ii in range(len(images)):
        t = images[ii]
        t_name = t.getAttribute("name").split('/')[-1]
        t_width = int(t.getAttribute("width"))
        t_height = int(t.getAttribute("height"))    

        label_file = open(os.path.join(txt_save_path,t_name[:11]+'.txt'),'w')

        try:
            img_name = [n for n in os.listdir(src_path) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_path,img_name)).convert('RGB')
        except:
            # sub_folder = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path,f))][0]
            img_name = [n for n in os.listdir(os.path.join(src_path, 'Twinning wisp')) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_path, 'Twinning wisp', img_name)).convert('RGB')

        nt = -1
        t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
        for t_ply in t_poly:
            nt = nt + 1
            t_ply_label = t_ply.getAttribute("label").split('_Reflection')[0]

            assert t_ply_label in class_list.keys()

            t_img = Image.new('L',(t_width,t_height))
            draw = ImageDraw.Draw(t_img)

            t_ply_points = t_ply.getAttribute("points")
            t_ply_points = t_ply_points.split(";")
            t_ply_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_ply_points])
            t_x = t_ply_points[:,0]
            t_y = t_ply_points[:,1]
            t_xy = [(x,y) for x,y in zip(t_x,t_y)]

            draw.polygon(t_xy, fill='white')
            # draw.line(t_xy, fill='white', width=2)

            t_img_np = np.array(t_img)
            lbl = label(t_img_np)
            props = regionprops(lbl)
            if len(props)>1:
                draw.line(t_xy, fill='white', width=2)
                t_img_np = np.array(t_img)
                lbl = label(t_img_np)
                props = regionprops(lbl)
                if len(props)>1:
                    print(src_path,t_name,'\n' f'when inclusion {nt} {t_ply_label} should have only 1 region but got {len(props)}')
                    exit()
            if props[0].area<50 and not t_ply_label.lower() == 'pinpoint':
                print(t_name, f'{nt} should be pinpoint but got {t_ply_label}')
                exit()

            t_class_id = class_list[t_ply_label]
            print(t_name[:11]+'_'+str(nt)+'.png',t_class_id, t_ply_label)
            label_file.write('{} {} {}\n'.format(t_name[:11]+'_'+str(nt)+'.png',t_class_id, t_ply_label))

            prop = props[0]
            box = (prop.bbox[1]-clear_dist, prop.bbox[0]-clear_dist, prop.bbox[3]+clear_dist, prop.bbox[2]+clear_dist)
            img_tmp = img.crop(box)
            if resize:
                img_tmp = img_tmp.resize(tar_size,Image.NEAREST)
            os.makedirs(os.path.join(save_path,t_ply_label),exist_ok=True)
            img_tmp.save(os.path.join(save_path,t_ply_label,t_name[:11]+'_'+str(nt)+'.png'))

        label_file.close()


def CutAndLabel_true_inclusion(xmlpath,src_path,save_path,txt_save_path,resize=True,tar_size=(32,32),clear_dist=2):
    dom = xml.dom.minidom.parse(os.path.join(src_path,xmlpath))
    xml_root = dom.documentElement
    images = xml_root.getElementsByTagName('image')
    
    # set multiclass 'Pinpoint':1,'
    all_class = {'Crystal':2,'Needle':3,'Cloud':4, 'Twinning_wisp':5,'Feather':6, 
                 'Internal_graining':7,'Nick':8,'Pit':9,'Burn_mark':10}    

    for ii in range(len(images)):
        t = images[ii]
        t_name = t.getAttribute("name").split('/')[-1]
        t_width = int(t.getAttribute("width"))
        t_height = int(t.getAttribute("height"))    
        label_file = open(os.path.join(txt_save_path,t_name[:11]+'.txt'),'w')
    
        try:
            img_name = [n for n in os.listdir(src_path) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_path,img_name)).convert('RGB')
        except:
            sub_folder = [f for f in os.listdir(src_path) if os.path.isdir(os.path.join(src_path,f))][0]
            img_name = [n for n in os.listdir(os.path.join(src_path, sub_folder)) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_path, sub_folder, img_name)).convert('RGB')
    
        nt = 0
        t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
        for t_ply in t_poly:
            t_ply_label = t_ply.getAttribute("label")
            if 'Reflection' in t_ply_label or not t_ply_label.split('_')[0] in all_class.keys():
                continue
            
            t_img = Image.new('L',(t_width,t_height))
            draw = ImageDraw.Draw(t_img)
            
            t_ply_points = t_ply.getAttribute("points")
            t_ply_points = t_ply_points.split(";")
            t_ply_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_ply_points])
            t_x = t_ply_points[:,0]
            t_y = t_ply_points[:,1]
            t_xy = [(x,y) for x,y in zip(t_x,t_y)]

            draw.polygon(t_xy, fill='white')
            # draw.line(t_xy, fill='white', width=2)
    
            t_img_np = np.array(t_img)
            lbl = label(t_img_np)
            props = regionprops(lbl)
            if len(props)>1 or props[0].area<50:
                print(src_path,t_name)
                break

            t_class_id = all_class[t_ply_label]
            print(t_name[:11]+'_'+str(nt)+'.png',t_class_id)
            label_file.write('{} {}\n'.format(t_name[:11]+'_'+str(nt)+'.png',t_class_id))

            prop = props[0]
            box = (prop.bbox[1]-clear_dist, prop.bbox[0]-clear_dist, prop.bbox[3]+clear_dist, prop.bbox[2]+clear_dist)
            img_tmp = img.crop(box)
            if resize:
                img_tmp = img_tmp.resize(tar_size,Image.NEAREST)
            os.makedirs(os.path.join(save_path,t_ply_label),exist_ok=True)
            img_tmp.save(os.path.join(save_path,t_ply_label,t_name[:11]+'_'+str(nt)+'.png'))
    
            nt = nt + 1

        label_file.close()


def main():
    base_path = '/media/hdd/diamond_data/hdr_data_v2'
    # base_path = '20210129 65 HDR images and videos (1920 - 1984)/20210119 CityU VVS2, VS2 to I1 (C)(id216-214)'
    save_path = '/media/hdd/diamond_data/cls_multi-class_EfficientNet'
    txt_save_path = os.path.join(save_path,'txt_file')
    os.makedirs(save_path,exist_ok=True)
    os.makedirs(txt_save_path,exist_ok=True)

    # save patches with inclusion at center
    for root,dirs,files in os.walk(base_path):
        status = checkXML(files)
        if status is not None:
            print(root)
            CutAndLabel(status,root,save_path,txt_save_path,resize=False,tar_size=(128,128),clear_dist=10)
            # CutAndLabel_true_inclusion(status,root,save_path,txt_save_path,resize=False,tar_size=(64,64),clear_dist=10)


if __name__ == "__main__":
	main()







