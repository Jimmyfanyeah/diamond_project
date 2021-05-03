# Generate masks according xml file
import xml.dom.minidom
import numpy as np
from PIL import Image, ImageDraw
import os
from skimage.segmentation import mark_boundaries as mkbdy

def genMaskImage(targetFileName,saveDir):
	saveFolder = saveDir
	os.makedirs(saveFolder,exist_ok=True)
	dom = xml.dom.minidom.parse(targetFileName)
	root = dom.documentElement

	images = root.getElementsByTagName('image')

	for t in images:
		t_name = t.getAttribute("name").split('/')[-1]
		t_width = int(t.getAttribute("width"))
		t_height = int(t.getAttribute("height"))

		t_img = Image.new('RGB',(t_width,t_height))
		draw = ImageDraw.Draw(t_img)

		t_polygon = t.getElementsByTagName("polygon")
		for t_plg in t_polygon:
			t_plg_points = t_plg.getAttribute("points")
			t_plg_points = t_plg_points.split(";")
			t_plg_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_plg_points])
			t_x = t_plg_points[:,0]
			t_y = t_plg_points[:,1]
			t_xy = [(x,y) for x,y in zip(t_x,t_y)]
			draw.polygon(t_xy, fill='white')
			draw.line(t_xy, fill='white', width=2)

		t_polyline = t.getElementsByTagName("polyline")
		for t_pl in t_polyline:
			t_pl_points = t_pl.getAttribute("points")
			t_pl_points = t_pl_points.split(";")
			t_pl_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_pl_points])
			t_x = t_pl_points[:,0]
			t_y = t_pl_points[:,1]
			t_xy = [(x,y) for x, y in zip(t_x, t_y)]
			draw.polygon(t_xy, fill='white')
			draw.line(t_xy, fill='white', width=2)

		t_img.save(os.path.join(saveFolder, t_name[:11]+'-mask.png'))
		# print(t_name[:11] + ' saved ...')


def MarkBoundary(targetFolderName,targetFileName,saveDir):
	saveFolder = saveDir
	os.makedirs(saveFolder,exist_ok=True)
	dom = xml.dom.minidom.parse(os.path.join(targetFolderName,targetFileName))
	root = dom.documentElement

	images = root.getElementsByTagName('image')

	for t in images:
		t_name = t.getAttribute("name").split('/')[-1]
		t_width = int(t.getAttribute("width"))
		t_height = int(t.getAttribute("height"))

		t_img = Image.new('L',(t_width,t_height))
		draw = ImageDraw.Draw(t_img)

		t_polygon = t.getElementsByTagName("polygon")
		for t_plg in t_polygon:
			t_plg_points = t_plg.getAttribute("points")
			t_plg_points = t_plg_points.split(";")
			t_plg_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_plg_points])
			t_x = t_plg_points[:,0]
			t_y = t_plg_points[:,1]
			t_xy = [(x,y) for x,y in zip(t_x,t_y)]
			draw.polygon(t_xy, fill='white')
			draw.line(t_xy, fill='white', width=2)

		t_polyline = t.getElementsByTagName("polyline")
		for t_pl in t_polyline:
			t_pl_points = t_pl.getAttribute("points")
			t_pl_points = t_pl_points.split(";")
			t_pl_points = np.asarray([[int(float(b)) for b in a.split(",")] for a in t_pl_points])
			t_x = t_pl_points[:,0]
			t_y = t_pl_points[:,1]
			t_xy = [(x,y) for x, y in zip(t_x, t_y)]
			draw.polygon(t_xy, fill='white')
			draw.line(t_xy, fill='white', width=2)

			t_img_np = np.array(t_img)
			try:
				img_name = [n for n in os.listdir(targetFolderName) if t_name[:11] in n and 'png' in n][0]
				img = Image.open(os.path.join(targetFolderName,img_name)).convert('RGB')
			except:
				img_name = [n for n in os.listdir(os.path.join(targetFolderName,'Twinning wisp')) if t_name[:11] in n and 'png' in n][0]
				img = Image.open(os.path.join(targetFolderName,'Twinning wisp',img_name)).convert('RGB')
			img_np = np.array(img)
			# print(img_np.shape,t_img_np[:,:,0].shape)
			img_np_mask = mkbdy(img_np,t_img_np[:,:],color=(1,0,0),outline_color=(1,0,0))
			img_double = Image.fromarray((img_np_mask*255).astype('uint8'))
			img_double.save(os.path.join(saveFolder, t_name[:11]+'-mask.png'))
			# print(t_name[:11] + ' saved ...')
			# except:
			# 	print(t_name)
			# 	print(targetFolderName)
