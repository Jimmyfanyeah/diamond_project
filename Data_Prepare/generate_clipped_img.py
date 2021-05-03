# generate 36 200x200 images from one 1200x1200 image
import numpy as np 
from PIL import Image 
import os

def handle_image(filename,label_filename,src_folder,save_folder,tar_size=(400,400),stride=200):
	img = Image.open(os.path.join(src_folder, filename))
	H, W = img.size 
	if img.mode != 'RGB':
		img = img.convert('RGB')

	for ii in range(int((H-tar_size[0])/stride)+1):
		print(ii)
		for jj in range(int((W-tar_size[1])/stride)+1):
			rect = (jj*stride, ii*stride, jj*stride+tar_size[1], ii*stride+tar_size[0])
			tmp_name = filename[:11]+'_'+str(ii)+'_'+str(jj)

			try:
				mask_img = Image.open(os.path.join(src_folder, label_filename))
			except IOError:
				mask_img = Image.new('RGB', tar_size)
			mask_img = mask_img.crop(rect)
			mask_img.save(os.path.join(save_folder, tmp_name+'-mask.png'))

			region = img.crop(rect)
			region.save(os.path.join(save_folder, tmp_name+'.png'))


def main():
	f = open(os.path.join(src_folder, "id.txt"))

	for line in f:
		filename = line[:-1]
		label_filename = filename[:11] + "-mask.png"
		# try:
		print(filename)
		handle_image(filename, label_filename, save_folder=save_folder)
		# except IOError:

	f.close()

if __name__ == "__main__":
	main()