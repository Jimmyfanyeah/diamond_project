# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:01:04 2021
@author: mastaffs

Target: mark out inclusions on original images
"""
import os
import numpy as np
import xml.dom.minidom
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import mark_boundaries as mkbdy
from shutil import copy2


def checkXML(filelist):
    status = None
    for f in filelist:
        if f.split('.')[-1] == 'xml':
            status = f
    return status


######## Mark out multi-class, types of inclusions, inclusion and reflection not distinguished
def MarkBoundary_inclusion(src_folder,xmlfile,save_folder,is_index=False,fontsize=5):
    # categorys = {'Crystal':0, 'Cloud':1, 'Pinpoint':2, 'Feather':3, 'Twinning_wisp':4, 'Needle':5}
    # corresponding color for types of inclusions
    colors = {'Pinpoint':(255,0,0), 'Crystal':(0,255,0), 'Needle':(0,0,255), 
              'Cloud':(255,0,0), 'Twinning_wisp':(0,255,0), 'Feather':(0,0,255),
              'Internal_graining':(255,0,0)}

    # used for mark boundary & legend
    colors_mkbdy = [(1,0,0),(0,1,0),(0,0,1),
                    (1,1,0),(1,0,1),(1,0.5098,0.2784),
                    (0.5451,0.2706,0.0745),(1,0.4157,0.4157),(1,0.7569,0.1451)]
    
    mask1 = ['Pinpoint', 'Crystal', 'Needle']
    mask2 = ['Cloud', 'Twinning_wisp', 'Feather']
    mask3 = ['Internal_graining']
    
    dom = xml.dom.minidom.parse(os.path.join(src_folder,xmlfile))
    xml_root = dom.documentElement
    images = xml_root.getElementsByTagName('image')
    
    for ii in range(len(images)):
        t = images[ii]
        t_width = int(t.getAttribute("width"))
        t_height = int(t.getAttribute("height"))
        t_name = t.getAttribute("name").split('/')[-1]
        
        img_mask1 = Image.new('RGB', (int(t_width), int(t_height)))
        draw1 = ImageDraw.Draw(img_mask1)
        img_mask2 = Image.new('RGB', (int(t_width), int(t_height)))
        draw2 = ImageDraw.Draw(img_mask2)
        img_mask3 = Image.new('RGB', (int(t_width), int(t_height)))
        draw3 = ImageDraw.Draw(img_mask3)
        
        t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
        for t_ply in t_poly:
            t_ply_points = t_ply.getAttribute("points")
            t_ply_points = t_ply_points.split(";")
            t_ply_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_ply_points])
            t_x = t_ply_points[:,0]
            t_y = t_ply_points[:,1]
            t_xy = [(x,y) for x,y in zip(t_x,t_y)]
            # t_ply_label = t_ply.getAttribute("label").split('_')[0]
            t_ply_label = t_ply.getAttribute("label")

            if t_ply_label in mask1:
                draw1.polygon(t_xy, fill=colors[t_ply_label])
            elif t_ply_label in mask2:
                draw2.polygon(t_xy, fill=colors[t_ply_label])
            elif t_ply_label in mask3:
                draw3.polygon(t_xy, fill=colors[t_ply_label])

        mask1_np, mask2_np, mask3_np = np.array(img_mask1), np.array(img_mask2), np.array(img_mask3)
        mask_np = np.concatenate((mask1_np,mask2_np,mask3_np),axis=2)

        try:
            img_name = [n for n in os.listdir(src_folder) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_folder,img_name)).convert('RGB').resize((t_width,t_height),Image.LANCZOS)
        except:
            img_name = [n for n in os.listdir(os.path.join(src_folder,'Twinning wisp')) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_folder,'Twinning wisp',img_name)).convert('RGB').resize((t_width,t_height),Image.LANCZOS)

        img_np = np.array(img)
        img_mask = img_np
        for ii in range(mask_np.shape[2]):
            img_mask = mkbdy(img_mask,mask_np[:,:,ii],color=colors_mkbdy[ii],outline_color=colors_mkbdy[ii])
        img_mask = Image.fromarray((img_mask*255).astype('uint8'))
    
        if is_index:
            defect_idx = 0
            font = ImageFont.truetype("arial.ttf", fontsize)
            draw_index = ImageDraw.Draw(img_mask)
            
            t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
            for t_ply in t_poly:
                t_ply_points = t_ply.getAttribute("points")
                t_ply_points = t_ply_points.split(";")
                t_ply_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_ply_points])
                t_x = t_ply_points[:,0]
                t_y = t_ply_points[:,1]
                t_xy = [(x,y) for x,y in zip(t_x,t_y)]
                draw_index.text(t_xy[-1], str(defect_idx),font=font,align ="left")   
                defect_idx = defect_idx + 1

        img_mask.save(os.path.join(save_folder, t_name[:11]+'_inclusion.png'))
        print(t_name[:11] + ' saved ...')


######## Mark out 2 class, inclusion (red) and reflection (green)
def MarkBoundary_inclusion_reflection(src_folder,xmlfile,save_folder,is_index=False,fontsize=5):
    # corresponding color for inclusion and reflection
    # reflection = (255,0,0), inclusion = (0,255,0)
    colors_mkbdy = [(1,0,0),(0,1,0),(0,0,1)]

    dom = xml.dom.minidom.parse(os.path.join(src_folder,xmlfile))
    xml_root = dom.documentElement
    images = xml_root.getElementsByTagName('image')
    
    for ii in range(len(images)):
        t = images[ii]
        t_width = int(t.getAttribute("width"))
        t_height = int(t.getAttribute("height"))
        t_name = t.getAttribute("name").split('/')[-1]
        
        img_mask = Image.new('RGB', (int(t_width), int(t_height)))
        draw = ImageDraw.Draw(img_mask)
        
        t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
        for t_ply in t_poly:
            t_ply_points = t_ply.getAttribute("points")
            t_ply_points = t_ply_points.split(";")
            t_ply_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_ply_points])
            t_x = t_ply_points[:,0]
            t_y = t_ply_points[:,1]
            t_xy = [(x,y) for x,y in zip(t_x,t_y)]
            t_ply_label = t_ply.getAttribute("label")

            if not 'Reflection' in t_ply_label:
                draw.polygon(t_xy, fill=(0,255,0))
            else:
                draw.polygon(t_xy, fill=(255,0,0))

        mask_np = np.array(img_mask)
        try:
            img_name = [n for n in os.listdir(src_folder) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_folder,img_name)).convert('RGB').resize((t_width,t_height),Image.LANCZOS)
        except:
            img_name = [n for n in os.listdir(os.path.join(src_folder,'Twinning wisp')) if t_name[:11] in n and 'png' in n][0]
            img = Image.open(os.path.join(src_folder,'Twinning wisp',img_name)).convert('RGB').resize((t_width,t_height),Image.LANCZOS)

        img_np = np.array(img)
        img_mask = img_np
        for ii in range(mask_np.shape[2]):
            img_mask = mkbdy(img_mask,mask_np[:,:,ii],color=colors_mkbdy[ii],outline_color=colors_mkbdy[ii])
        img_mask = Image.fromarray((img_mask*255).astype('uint8'))
    
        if is_index:
            defect_idx = 0
            font = ImageFont.truetype("arial.ttf", fontsize)
            draw_index = ImageDraw.Draw(img_mask)
            
            t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
            for t_ply in t_poly:
                t_ply_points = t_ply.getAttribute("points")
                t_ply_points = t_ply_points.split(";")
                t_ply_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_ply_points])
                t_x = t_ply_points[:,0]
                t_y = t_ply_points[:,1]
                t_xy = [(x,y) for x,y in zip(t_x,t_y)]
                draw_index.text(t_xy[-1], str(defect_idx),font=font,align ="left")   
                defect_idx = defect_idx + 1

        img_mask.save(os.path.join(save_folder, t_name[:11]+'_inclusion_reflection.png'))
        print(t_name[:11] + ' saved ...')


def main():
    base_path = r'F:\diamond_project\hdr_data_v2'
    save_path = r'F:\diamond_project\visualization\true_inclusion_classfication'
    # save_path = r'F:\diamond_project\visualization\all_inclusion_reflection'
    os.makedirs(save_path,exist_ok=True)
    
    ########### save visualization images
    for root,dirs,files in os.walk(base_path):
        status = checkXML(files)
        if status is not None:
            print(root)
            MarkBoundary_inclusion(root,status,save_path,is_index=True,fontsize=12)
            # MarkBoundary_inclusion_reflection(root,status,save_path,is_index=True,fontsize=12)
    
    ############ generate legend for method
    # MarkBoundary_inclusion & MarkBoundary_inclusion_reflection
    type = 'MarkBoundary_inclusion'  
    if type == 'MarkBoundary_inclusion':
        colors = {'Pinpoint':(255,0,0), 'Crystal':(0,255,0), 'Needle':(0,0,255),
              'Cloud':(255,255,0), 'Twinning_wisp':(255,0,255), 'Feather':(255,129,70),
              'Internal_graining':(139,69,18)}
    elif type == 'MarkBoundary_inclusion_reflection':
        colors = {'Inclusion':(255,0,0),'Reflection':(0,255,0)}
    legend = Image.new('RGB', (200,200))
    draw_legend = ImageDraw.Draw(legend)
    row_idx = 10
    
    for key in colors.keys():
        draw_legend.text((30,row_idx),key,colors[key])
        row_idx = row_idx+10
    legend.save(os.path.join(save_path,'legend.png'))
    
    ############# copy all images into folder
    base_path = r'F:\diamond_project\hdr_data_v2'
    save_path = r'F:\diamond_project\visualization/all_imgs'
    os.makedirs(save_path,exist_ok=True)
    for root,dirs,files in os.walk(base_path):
        print(root)
        imgList = [n for n in files if 'png' in n and 'mask' not in n]
        for n in imgList:
            copy2(os.path.join(root,n),os.path.join(save_path,n))
    
if __name__ == "__main__":
	main()     
            


import os
import numpy as np
from PIL import Image
from skimage.segmentation import mark_boundaries as mkbdy

base_path = r'D:\OneDrive - City University of Hong Kong\1214'

mask = Image.open(os.path.join(base_path,'10345832276_mask.png'))
img_path = os.path.join(base_path, '10345832276.png')

mask_np = np.array(mask)
img = Image.open(img_path).convert('RGB')
img_np = np.array(img)

img_np = mkbdy(img_np, mask_np[:,:,0], color=(1,0,0), outline_color=(1,0,0))
img_np = mkbdy(img_np, mask_np[:,:,1], color=(0,1,0), outline_color=(0,1,0))
img_np = Image.fromarray((img_np*255).astype('uint8'))

img_np.save(os.path.join(base_path,'10345832276_mask_on_img.png'))







            
            