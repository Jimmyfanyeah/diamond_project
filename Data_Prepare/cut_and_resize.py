# clip the images to remove all crust
# resize the resulting image and corresponding label into 1200 by 1200 RGB file
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


def handle_image(filename,label_filename,src_folder,save_folder,is_resize=True,tar_size=(1200,1200)):

	file = open(os.path.join(save_folder,'cut_locations.txt'),'a')
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
	file.write('{},{},{},{},{}\n'.format(filename.split('.')[0],left, top, right, bottom))
	region = img.crop(rect)
	if is_resize:
		region = region.resize(tar_size,Image.LANCZOS)
	region.save(os.path.join(save_folder, filename))

	try:
		mask_img = Image.open(os.path.join(src_folder, label_filename))
	except IOError: # If no mask, then the diamond has no inclusion/reflection.
		mask_img = Image.new('RGB',(width,height),color=0)
	mask_img = mask_img.crop(rect)
	if is_resize:
		mask_img = mask_img.resize(tar_size,Image.NEAREST)
	mask_img.save(os.path.join(save_folder, label_filename))
	file.close()


def handle_image_test(filename,label_filename,src_folder,save_folder,is_resize=True,tar_size=(1200,1200)):
    
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
	if is_resize:
		region = region.resize(tar_size,Image.LANCZOS)
	region.save(os.path.join(save_folder, filename))
	try:
		mask_img = Image.open(os.path.join(src_folder, label_filename))
	except IOError: # If no mask, then the diamond has no inclusion/reflection.
		mask_img = Image.new('RGB',(width,height),color=0)

	mask_img = mask_img.crop(rect)
	if is_resize:
		mask_img = mask_img.resize(tar_size,Image.NEAREST)
	mask_img.save(os.path.join(save_folder, label_filename))
	file.close()



def main():

	f = open("/Users/zitongwang/Desktop/2020/Chow_Proj/processed_data/with_labels.txt","r")
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