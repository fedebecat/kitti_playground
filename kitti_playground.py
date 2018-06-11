import cv2
import matplotlib.pyplot as plt
from random import shuffle
from kitti_utils import KittiAnnotation

def get_colormap(N):
    cm = plt.cm.get_cmap(None, N)
    colors = [cm(x) for x in range(N)]
    for i in range(annot.num_tracks):
        colors[i] = [x * 255 for x in colors[i][:3]]
    # randomize colormap
    shuffle(colors)
    return colors


video_id = '0020'
file_path = '/media/becattini/399B724D60527D8A/dataset/kitti/raw_data_downloader/training/label_02/' + video_id + '.txt'
img_path = '/media/becattini/399B724D60527D8A/dataset/kitti/raw_data_downloader/training/image_02/' + video_id
annot = KittiAnnotation(file_path, img_path)
gen = annot.annot_generator(data=('img', 'annot'), loop=False)  # use loop=True to loop indefinitely over the video

FPS = 30
colors = get_colormap(annot.num_tracks)

while True:
    img, boxes = gen.next()
    if img is None:
        break
    for track_id in boxes.keys():
        box, obj_type = boxes[track_id]
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=colors[int(track_id)], thickness=2)
        cv2.putText(img, obj_type, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[int(track_id)], 2)

    cv2.imshow('img', img)
    cv2.waitKey(1000/FPS)
