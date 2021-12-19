from __future__ import print_function
import cv2
import os
import numpy as np
from PIL import Image
from numpy import asarray
import math
from MyUtil import ResizeWithAspectRatio, my_ssim
import shutil


def track_defect_on_diamond(fileid, src_folder,save_folder,video_folder):

    # Initialize saving folders and files
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'all_inclusion'), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'all_reflection'), exist_ok=True)

    save_txt_folder = os.path.join(save_folder, str(fileid) + '_trajectory')
    if os.path.isdir(save_txt_folder):
        shutil.rmtree(save_txt_folder)
    os.makedirs(save_txt_folder)

    # Initialize parameters
    screen_size = 800
    bounding_boxes = []
    box_width = 3
    temp_file_count = 0
    template = []
    filename = []
    last_frames = []
    lf_or_template = []

    
    if not os.path.isfile(os.path.join(video_folder, fileid + '.mp4')):
        return None
    
    # Open the video file
    cap = cv2.VideoCapture(os.path.join(video_folder, fileid + '.mp4'))
    _, frame = cap.read()

    # Count number of of defects
    temp_folder = os.path.join(src_folder, fileid + '_template')
    files = [n for n in os.listdir(temp_folder) if n[-5:-4] == '1']
    # files = os.listdir(temp_folder)
    temp_file_count = len(files)
    temp_file_count /= 2
    colors = np.zeros((int(temp_file_count), 3))
    is_inclusion = np.zeros((int(temp_file_count),))

    # Read template segmentations of defects
    for i in range(int(temp_file_count)):
        # i = int(files[jj].split('_')[1])
        if os.path.isfile(os.path.join(temp_folder, fileid + '_' + str(i) + '_0.txt')):
            is_inclusion[i] = 0
            f = open(os.path.join(temp_folder, fileid + '_' + str(i) + '_0.txt'), 'r')
            template.append(Image.open(os.path.join(temp_folder + '/' + fileid + '_' + str(i) + '_0.png')))
            colors[i, :] = (255, 0, 0)
        if os.path.isfile(os.path.join(temp_folder, fileid + '_' + str(i) + '_1.txt')):
            is_inclusion[i] = 1
            f = open(os.path.join(temp_folder, fileid + '_' + str(i) + '_1.txt'), 'r')
            template.append(Image.open(os.path.join(temp_folder + '/' + fileid + '_' + str(i) + '_1.png')))
            colors[i, :] = (0, 0, 255)

        # clear all files before
        if is_inclusion[i] > 0:
            try:
                os.remove(os.path.join(save_folder, 'all_inclusion', fileid + '_' + str(i) + '_1.txt'))
                print(f'{fileid} exist, delete and re-run now')
            except:
                pass
        else:
            try:
                os.remove(os.path.join(save_folder, 'all_reflection', fileid + '_' + str(i) + '_0.txt'))
                print(f'{fileid} exist, delete and re-run now')
            except:
                pass
        
        if is_inclusion[i] > 0:
            bbox = f.readline()
            bbox = tuple(map(int, bbox.split(' ')))
            template[i] = asarray(template[i])
            template[i] = cv2.cvtColor(template[i], cv2.COLOR_RGB2GRAY)
            last_frames.append(frame[bbox[1]:bbox[1] + bbox[3] + 1, bbox[0]:bbox[0] + bbox[2] + 1])
            last_frames[i] = asarray(last_frames[i])
            last_frames[i] = cv2.cvtColor(last_frames[i], cv2.COLOR_RGB2GRAY)
            bounding_boxes.append([])
            bounding_boxes[i].append(bbox)
            if bbox[2] * bbox[3] > 500:
                lf_or_template.append(np.array(1))
            else:
                lf_or_template.append(np.array(2))
            filename = open(
                os.path.join(save_folder, fileid+'_trajectory', fileid+'_'+str(i)+'_'+ str(int(is_inclusion[i]))+'.txt'),
                'w')
            filename.write(
                str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])))
            filename.write('\n')
            
            if is_inclusion[i] > 0:
                copy_filename = open(os.path.join(save_folder, 'all_inclusion', fileid + '_' + str(i) + '_1.txt'), 'a+')
            else:
                copy_filename = open(os.path.join(save_folder, 'all_reflection', fileid + '_' + str(i) + '_0.txt'), 'a+')
            copy_filename.write(
                str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])))
            copy_filename.write('\n')

    # Process video and track objects (ssim tracker)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        for k in range(int(temp_file_count)):
            # Compute ssim
            similarity = np.zeros((box_width * 2 + 1, box_width * 2 + 1))
            for i in range(-box_width, box_width + 1):
                for j in range(-box_width, box_width + 1):
                    current_box = frame[
                                  bounding_boxes[k][-1][1] + i:bounding_boxes[k][-1][1] + i + bounding_boxes[k][-1][3]
                                                               + 1,
                                  bounding_boxes[k][-1][0] + j:bounding_boxes[k][-1][0] + j
                                                               + bounding_boxes[k][-1][2] + 1]
                    current_box = cv2.cvtColor(current_box, cv2.COLOR_RGB2GRAY)
                    if lf_or_template[k] == 1:
                        similarity[i + box_width, j + box_width] = my_ssim(last_frames[k], current_box, radius=1,
                                                                       neighborhood=5, window_size=5, alpha=1, beta=1,
                                                                       gamma=1)
                    else:
                        similarity[i + box_width, j + box_width] = my_ssim(template[k], current_box, radius=1,
                                                                           neighborhood=7, window_size=5, alpha=1.2,
                                                                           beta=1,
                                                                           gamma=1)
            # Extract the box with maximum ssim
            row = np.argmax(similarity) % similarity.shape[1] - box_width
            col = math.floor(np.argmax(similarity) / similarity.shape[1]) - box_width
            bbox = (bounding_boxes[k][-1][0] + row, bounding_boxes[k][-1][1] + col, bounding_boxes[k][-1][2],
                    bounding_boxes[k][-1][3])
            bounding_boxes[k].append(bbox)
            last_frames[k] = asarray(frame[bbox[1]:bbox[1] + bbox[3] + 1, bbox[0]:bbox[0] + bbox[2] + 1])
            last_frames[k] = cv2.cvtColor(last_frames[k], cv2.COLOR_RGB2GRAY)

            # Print results
            print(bounding_boxes[k].__len__(), bbox, np.max(similarity))
            filename = open(os.path.join(save_folder, fileid + '_trajectory/' + fileid + '_' + str(k) + '_' + str(
                int(is_inclusion[k])) + '.txt'), 'a+')
            filename.write(
                str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])))
            filename.write('\n')
            if is_inclusion[k] > 0:
                copy_filename = open(os.path.join(save_folder, 'all_inclusion/' + fileid + '_' + str(k) + '_1.txt'),
                                     'a+')
            else:
                copy_filename = open(os.path.join(save_folder, 'all_reflection/' + fileid + '_' + str(k) + '_0.txt'),
                                     'a+')
            copy_filename.write(
                str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])))
            copy_filename.write('\n')

            # Draw the box on the display frame
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, colors[k, :], 2, 1)
            cv2.putText(frame, str(k), p1, cv2.FONT_HERSHEY_PLAIN, 1, colors[k, :], 1, cv2.LINE_AA)

        # display the frame with boxes
        frame = ResizeWithAspectRatio(frame, width=screen_size)
        cv2.imshow('MultiTracker', frame)

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    # Re-open the files
    bounding_boxes = []
    cap = cv2.VideoCapture(os.path.join(video_folder, fileid + '.mp4'))
    for i in range(int(temp_file_count)):
        if os.path.isfile(os.path.join(temp_folder, fileid + '_' + str(i) + '_0.txt')):
            f = open(os.path.join(temp_folder, fileid + '_' + str(i) + '_0.txt'), 'r')
        else:
            f = open(os.path.join(temp_folder, fileid + '_' + str(i) + '_1.txt'), 'r')
        bbox = f.readline()
        bbox = tuple(map(int, bbox.split(' ')))
        bounding_boxes.append([])
        bounding_boxes[i].append(bbox)

    # Process video and track objects (ssim tracker)
    for ii in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - ii - 1)
        success, frame = cap.read()
        if not success:
            break
        for k in range(int(temp_file_count)):
            # Compute ssim
            similarity = np.zeros((box_width * 2 + 1, box_width * 2 + 1))
            for i in range(-box_width, box_width + 1):
                for j in range(-box_width, box_width + 1):
                    current_box = frame[
                                  bounding_boxes[k][-1][1] + i:bounding_boxes[k][-1][1] + i + bounding_boxes[k][-1][3]
                                                               + 1,
                                  bounding_boxes[k][-1][0] + j:bounding_boxes[k][-1][0] + j
                                                               + bounding_boxes[k][-1][2] + 1]
                    current_box = cv2.cvtColor(current_box, cv2.COLOR_RGB2GRAY)
                    if lf_or_template[k] == 1:
                        similarity[i + box_width, j + box_width] = my_ssim(last_frames[k], current_box, radius=1,
                                                                           neighborhood=5, window_size=5, alpha=1, beta=1,
                                                                           gamma=1)
                    else:
                        similarity[i + box_width, j + box_width] = my_ssim(template[k], current_box, radius=1,
                                                                           neighborhood=7, window_size=5, alpha=1.2,
                                                                           beta=1,
                                                                           gamma=1)
            # Extract the box with maximum ssim
            row = np.argmax(similarity) % similarity.shape[1] - box_width
            col = math.floor(np.argmax(similarity) / similarity.shape[1]) - box_width
            bbox = (bounding_boxes[k][-1][0] + row, bounding_boxes[k][-1][1] + col, bounding_boxes[k][-1][2],
                    bounding_boxes[k][-1][3], 0)
            bounding_boxes[k].append(bbox)

            # Print results
            print(bounding_boxes[k].__len__(), bbox, np.max(similarity))
            filename = open(os.path.join(save_folder, fileid + '_trajectory', fileid + '_' + str(k) + '_' + str(
                int(is_inclusion[k])) + '.txt'), 'a+')
            filename.write(
                str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])))
            filename.write('\n')
            if is_inclusion[k] > 0:
                copy_filename = open(os.path.join(save_folder, 'all_inclusion', fileid + '_' + str(k) + '_1.txt'),
                                     'a+')
            else:
                copy_filename = open(os.path.join(save_folder, 'all_reflection', fileid + '_' + str(k) + '_0.txt'),
                                     'a+')
            copy_filename.write(
                str(int(bbox[0])) + ' ' + str(int(bbox[1])) + ' ' + str(int(bbox[2])) + ' ' + str(int(bbox[3])))
            copy_filename.write('\n')

            # Draw the box on the display frame
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, colors[k, :], 2, 1)
            cv2.putText(frame, str(k), p1, cv2.FONT_HERSHEY_PLAIN, 1, colors[k, :], 1, cv2.LINE_AA)

        # display the frame with boxes
        frame = ResizeWithAspectRatio(frame, width=screen_size)
        cv2.imshow('MultiTracker', frame)
        

        # quit on ESC button
        if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
            break

    # Close files
    filename.close()
    cap.release()
    
    cv2.waitKey(120)
    cv2.destroyAllWindows()


def main():
    idx = 591
    video_folder = r'F:\Diamond\Videos'
    
    src_folder = os.path.join( r'F:\Diamond\trajectory_1213\_'+str(idx), 'inclusion_bbox')
    save_folder = r'F:\Diamond\trajectory_1213\_'+str(idx)

    os.makedirs(save_folder, exist_ok=True)
    fileids = [n[:11] for n in os.listdir(src_folder)]
    exist_fileids = [n[:11] for n in os.listdir(save_folder) if 'trajectory' in n]
    fileids = list(set(fileids).difference(set(exist_fileids)))
    fileids.sort()
    
    for fileid in fileids:
        track_defect_on_diamond(fileid,src_folder,save_folder,video_folder)


if __name__ == "__main__":
	main()