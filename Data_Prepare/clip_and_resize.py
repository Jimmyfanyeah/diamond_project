import numpy as np 
from PIL import Image 
import os

def is_not_crust(pix):
	if sum(pix)==0 or sum(pix)==765:
		return False
	else: return True


def h_check(img, y, step=1):
	width = img.size[0]
	for x in range(0, width, step):
		if is_not_crust(img.getpixel((x,y))):
			return False
	return True


def w_check(img, x, step=1):
	height = img.size[1]
	for y in range(0, height, step):
		if is_not_crust(img.getpixel((x,y))):
			return False
	return True


def boundary_finder(img, crust_side, core_side, checker):
	if not checker(img, crust_side):
		return crust_side
	if checker(img, core_side):
		return core_side

	mid = (crust_side + core_side) // 2
	while mid != core_side and mid != crust_side:
		if checker(img, mid):
			crust_side = mid
		else: 
			core_side = mid
		mid = (crust_side + core_side) // 2
	return core_side


def handle_image(filename,label_filename,src_folder,save_folder,tar_size=(1200,1200)):

	file = open(os.path.join(save_folder,'cutted.txt'),'a')
	img = Image.open(os.path.join(src_folder, filename))
	if img.mode != "RGB":
		img = img.convert("RGB")
	width, height = img.size

	left = boundary_finder(img, 0, width/2, w_check)
	right = boundary_finder(img, width-1, width/2, w_check)
	top = boundary_finder(img, 0, height/2, h_check)
	bottom = boundary_finder(img, height-1, height/2, h_check)

	rect = (left, top, right, bottom)
	print(filename+'\t\t', rect, np.int32)
	file.write('{},{},{},{},{}'.format(filename.split('.')[0],left, top, right, bottom))
	file.write('\n')
	region = img.crop(rect)
	region = region.resize(tar_size,Image.LANCZOS)
	region.save(os.path.join(save_folder, filename))
	try:
		mask_img = Image.open(os.path.join(src_folder, label_filename))
	except IOError: # If no mask, then the diamond has no inclusion/reflection.
		mask_img = Image.new('RGB',(width,height),color=0)
	
	mask_img = mask_img.crop(rect).resize(tar_size,Image.NEAREST)
	mask_img.save(os.path.join(save_folder, label_filename))
	file.close()



def handle_image_multi(filename,label_filename,src_folder,save_folder,tar_size=(1200,1200)):

	file = open(os.path.join(save_folder,'cutted.txt'),'a')
	img = Image.open(os.path.join(src_folder, filename))
	if img.mode != "RGB":
		img = img.convert("RGB")
	width, height = img.size

	left = boundary_finder(img, 0, width/2, w_check)
	right = boundary_finder(img, width-1, width/2, w_check)
	top = boundary_finder(img, 0, height/2, h_check)
	bottom = boundary_finder(img, height-1, height/2, h_check)

	rect = (left, top, right, bottom)
	print(filename+'\t\t', rect, np.int32)
	file.write('{},{},{},{},{}'.format(filename.split('.')[0],left, top, right, bottom))
	file.write('\n')
	region = img.crop(rect)
	region = region.resize(tar_size,Image.LANCZOS)
	region.save(os.path.join(save_folder, filename))
	file.close()

	for idx in range(3):
		label_filename = filename[:11]+'-mask_'+str(idx)+'.png'
		try:
			mask_img = Image.open(os.path.join(src_folder, label_filename))
		except IOError: # If no mask, then the diamond has no inclusion/reflection.
			mask_img = Image.new('RGB',(width,height),color=0)
		mask_img = mask_img.crop(rect).resize(tar_size,Image.NEAREST)
		mask_img.save(os.path.join(save_folder, label_filename))


def main():

	f = open("/home/lingjia/Documents/tmp/with_labels.txt","r")
	report_f = open(os.path.join(save_folder,'no_label_report.txt'), 'w')

	for line in f:
		filename = line[:-1]
		label_filename = filename[:11] + "-mask.png"
		try:
			handle_image(filename, label_filename, save_folder=save_folder)
		except IOError:
			report_f.write(filename)
			report_f.write('\n')

	f.close()
	report_f.close()

if __name__ == "__main__":
	main()