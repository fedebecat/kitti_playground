import cv2
import matplotlib.pyplot as plt
from random import shuffle
from kitti_utils import KittiAnnotation
import numpy as np


def get_colormap(N):
    cm = plt.cm.get_cmap(None, N)
    colors = [cm(x) for x in range(N)]
    for i in range(N):
        colors[i] = [x * 255 for x in colors[i][:3]]
    # randomize colormap
    shuffle(colors)
    return colors


def vis_mask(img, mask, col, alpha=0.4, show_border=True, border_thick=1):
    """Visualizes a single binary mask."""

    img = img.astype(np.float32)
    idx = np.nonzero(mask)

    img[idx[0], idx[1], :] *= 1.0 - alpha
    img[idx[0], idx[1], :] += alpha * (np.array(col)/255.0)

    if show_border:
        _, contours, _ = cv2.findContours(
            mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, (255, 255, 255), border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)


dataset_path = '/media/becattini/RedPro/dataset/kitti/'
video_id = '0000'
file_path = dataset_path + '/label_02/' + video_id + '.txt'
img_path = dataset_path + 'image_02/' + video_id
annot = KittiAnnotation(file_path, img_path)

'''
"gen" is a generator to obtain relevant data about each frame in the sequence 
use loop=True to loop indefinitely over the video
you can pass to data_from_generator the types you want the generator to yield
available types:
'img' -> the BGR frame
'annot' -> the ground truth data
'dets' -> Mask-RCNN detections
'masks' -> segmentation masks from Mask-RCNN
'''
data_from_generator = ('img', 'annot', 'dets')
gen = annot.annot_generator(data=data_from_generator, loop=True)

FPS = 30
colors_tracks = get_colormap(annot.num_tracks)

while True:
    cur_data = gen.next()
    img = cur_data['img']
    if img is None:
        break

    # annot
    if 'annot' in cur_data.keys():
        annot = cur_data['annot']
        for track_id in annot.keys():
            box, obj_type = annot[track_id]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=colors_tracks[int(track_id)], thickness=2)
            cv2.putText(img, obj_type, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_tracks[int(track_id)], 2)

    # dets
    if 'dets' in cur_data.keys():
        dets = cur_data['dets']
        for box, obj_type, score in zip(*dets):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=1)
            cv2.putText(img, obj_type + ': %.2f' % score, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # masks
    if 'masks' in cur_data.keys():
        masks = cur_data['masks']
        for mask in masks:
            img = vis_mask(img, mask, (0, 0, 255))

    cv2.imshow('img', img)
    cv2.waitKey(1000/FPS)