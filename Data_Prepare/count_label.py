import xml.dom.minidom
import numpy as np
from PIL import Image 
from PIL import ImageDraw
import os
from fnmatch import fnmatch
from collections import defaultdict

def checkXML(filelist):
	status = None
	for f in filelist:
		if f.split('.')[-1] == 'xml':
			status = f

	return status

types = []
count_dict = defaultdict(int)
src_folder = '/media/hdd/chow_data/hdr_data'
# src_folder = '/home/lingjia/Documents/hdr_data'
for root,dirs,files in os.walk(src_folder):
	status = checkXML(files)
	if status is not None:
		dom = xml.dom.minidom.parse(os.path.join(root,status))
		xml_root = dom.documentElement
		images = xml_root.getElementsByTagName('image')
		for t in images:
			t_polygon = t.getElementsByTagName("polygon")  
			for t_plg in t_polygon:
				t_plg_label = t_plg.getAttribute("label")
				count_dict[t_plg_label]=count_dict[t_plg_label]+1
				if t_plg_label not in types:
						types.append(t_plg_label)

			t_polyline = t.getElementsByTagName("polyline")
			for t_pl in t_polyline:
				t_pl_label = t_pl.getAttribute("label")
				count_dict[t_pl_label]=count_dict[t_pl_label]+1
				if t_pl_label not in types:
						types.append(t_pl_label)

print(count_dict)


# count id
# ids = []
# src_folder = '/home/lingjia/Documents/hdr_data'
# for root,dirs,files in os.walk(src_folder):
# 	status = checkXML(files)
# 	if 'Twinning wisp' not in root:
# 		if status is not None:
# 			dom = xml.dom.minidom.parse(os.path.join(root,status))
# 			xml_root = dom.documentElement
# 			images = xml_root.getElementsByTagName('image')
# 			for t in images:
# 				t_name = t.getAttribute("name").split('/')[-1]
# 				t_name = t_name[:11]
# 				if t_name not in ids:
# 					ids.append(t_name)
# 				else: print(t_name)
