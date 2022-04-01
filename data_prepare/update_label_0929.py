# change all defects and reflections smaller than 50 pixels = pinpoint
# pinpoint > 50 pixels = crystal
import os
import numpy as np
from PIL import Image, ImageDraw
import time
import xml.etree.ElementTree as ET

def checkXML(filelist):
    status = None
    for f in filelist:
        if f.split('.')[-1] == 'xml':
            status = f
    return status


def check_defect_size(xml_path,xml_file,selected_types,record_file):
    # TODO: change labels <50 pixels to pinpoint 

    tree = ET.parse(os.path.join(xml_path,xml_file))
    root = tree.getroot()
    images = root.findall('image')

    for t in images:
        defect_idx_pinpoint = []
        defect_idx_crystal = []
        defect_idx_change = []
        defect_idx = 0
        t_name = t.attrib['name']
        t_width = int(t.attrib['width'])
        t_height = int(t.attrib['height'])

        t_polys = t.findall('polygon') + t.findall("polyline")
        for t_poly in t_polys:
            defect_idx +=1
            t_poly_label = t_poly.attrib["label"]
            t_img = Image.new('L',(t_width,t_height))
            draw = ImageDraw.Draw(t_img)

            t_poly_points = t_poly.attrib["points"]
            t_poly_points = t_poly_points.split(";")
            t_poly_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_poly_points])
            t_x = t_poly_points[:,0]
            t_y = t_poly_points[:,1]
            t_xy = [(x,y) for x,y in zip(t_x,t_y)]

            draw.polygon(t_xy, fill='white')
            # draw.line(t_xy, fill='white', width=2)

            t_img_np = np.array(t_img)
            num_pixel = int(np.sum(t_img_np)/255)

            if num_pixel < 50 and t_poly_label not in selected_types:
                defect_idx_change.append(defect_idx)
                defect_idx_pinpoint.append(defect_idx)
                if 'Reflection' in t_poly_label:
                    t_poly.set('label','Pinpoint_Reflection')
                else: t_poly.set('label','Pinpoint')

            if num_pixel >= 50 and t_poly_label in selected_types:
                defect_idx_change.append(defect_idx)
                defect_idx_crystal.append(defect_idx)
                if 'Reflection' in t_poly_label:
                    t_poly.set('label','Crystal_Reflection')
                else: t_poly.set('label','Crystal')

        if len(defect_idx_change)>0:
            print(f'{t_name[:11]} pinpoint {defect_idx_pinpoint}')
            print(f'{t_name[:11]} crystal {defect_idx_crystal}')
            record_file.write(f'{t_name[:11]} {defect_idx_change}\n')

    tree.write(os.path.join(xml_path,xml_file))
    t = time.strftime("%Y%m%d", time.localtime())
    os.rename(os.path.join(xml_path,xml_file),os.path.join(xml_path,xml_file[:-4]+' rev'+t+'.xml'))


if __name__=='__main__':
    base_path = '/media/hdd/css_data/hdr_data_v2'

    selected_types = ['Pinpoint','Pinpoint_Reflection']
    record_file = open(os.path.join(base_path,'0929_record_change_label.txt'),'w')

    for root,dirs,files in os.walk(base_path):
        status = checkXML(files)
        if status is not None:
            print(root)
            record_file.write(root+'\n')
            check_defect_size(root,status,selected_types,record_file)

    record_file.close()











