from collections import defaultdict
import os
import xml.dom.minidom

def checkXML(filelist):
	status = None
	for f in filelist:
		if f.split('.')[-1] == 'xml':
			status = f
	return status

def count_inclusions(data_path):
	types = []
	count_dict = defaultdict(int)
	for root,dirs,files in os.walk(data_path):
		status = checkXML(files)
		if status is not None:
			dom = xml.dom.minidom.parse(os.path.join(root,status))
			xml_root = dom.documentElement
			images = xml_root.getElementsByTagName('image')
			for t in images:
				t_polygon = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
				for t_plg in t_polygon:
					t_plg_label = t_plg.getAttribute("label")
					count_dict[t_plg_label]=count_dict[t_plg_label]+1
					if t_plg_label not in types:
							types.append(t_plg_label)
	return count_dict

if __name__=='__main__':
    data_path = '/media/hdd/css_data/hdr_data_v2'
    count_dict = count_inclusions(data_path)
    sorted_dict = dict(sorted(count_dict.items()))
    print(sorted_dict)