import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as ttf
from torch.utils.data import DataLoader
import xml.dom.minidom
import os
from fnmatch import fnmatch
from collections import defaultdict

def checkXML(filelist):
	status = None
	for f in filelist:
		if f.split('.')[-1] == 'xml':
			status = f

	return status


if __name__ == '__main__':
    types = []
    count_dict = defaultdict(int)
    data_dir = '/media/hdd/diamond_data/hdr_data'
    for root,dirs,files in os.walk(data_dir):
        status = checkXML(files)
        if status is not None:
            count_dict_current = defaultdict(int)
            dom = xml.dom.minidom.parse(os.path.join(root,status))
            xml_root = dom.documentElement
            images = xml_root.getElementsByTagName('image')
            for t in images:
                t_poly = t.getElementsByTagName("polygon") + t.getElementsByTagName("polyline")
                for t_plg in t_poly:
                    t_plg_label = t_plg.getAttribute("label")
                    count_dict[t_plg_label] += 1
                    count_dict_current[t_plg_label] += 1
                    if t_plg_label not in types:
                            types.append(t_plg_label)

            print(f'root: {os.path.join(root,status)}')
            # print(f"{root} IG:{count_dict_current['Internal_graining']},IG_RE:{count_dict_current['Internal_graining_Reflection']}")

    print(count_dict)