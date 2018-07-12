# kitti_playground
A playground to test sequences from the kitti dataset http://www.cvlibs.net/datasets/kitti/

## Requirements
Download the tracking training sequences and labels from the [**kitti website**](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) or download a sample sequence from my [_google drive link_](https://drive.google.com/drive/folders/1YJxWvlZZ6vp9GQ5yT-dDtDFaxrDLcfTW?usp=sharing).
If you want Mask-RCNN detections and masks you can download the compressed pickle results [_here_](https://drive.google.com/drive/folders/1YJxWvlZZ6vp9GQ5yT-dDtDFaxrDLcfTW?usp=sharing)

## Libraries
 - Opencv
 - matplotlib
 - numpy
 - glob
 - ntpath
 - pickle
 - gzip

## Setting up `virtualenv`
```
virtualenv --system-site-packages kitti-env
source kitti-env/bin/activate
pip install -r requirements.txt
```
Add flag `--system-site-packages` to use `TKinter` for `matplotlib.plot`

## Run the code
Run _kitti_playground.py_ to see an example code

## Usage
```
usage: kitti_playground.py [-h] labels_path images_path video_id

positional arguments:
  labels_path  Directory containing the labels, e.g. label_02/
  images_path  Directory containing the images, e.g. image_02/
  video_id     Video ID, e.g. 0000

optional arguments:
  -h, --help   show this help message and exit

```

### Select video
Change _video_id_ to match the names of the videos (0000, 0001, 0002, ...)

### Select data
_gen_ is a generator to obtain relevant data about each frame in the sequence. You can use _loop=True_ to loop indefinitely over the video. You can pass to _data_from_generator_ the types you want the generator to yield.
Available types:
 - 'img' -> the BGR frame
 - 'annot' -> the ground truth data
 - 'dets' -> Mask-RCNN detections
 - 'masks' -> segmentation masks from Mask-RCNN
