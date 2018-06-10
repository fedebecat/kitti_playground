from sys import argv as cmdLineArgs
from xml.etree.ElementTree import ElementTree
import numpy as np
import itertools
from warnings import warn
import parseTrackletXML as xmlParser
from os.path import join, expanduser
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def draw3Dbox(trackletBox, ax=None, facecolor='cyan', edgecolor='red'):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    ax.scatter3D(trackletBox[0, :], trackletBox[1, :], trackletBox[2, :])

    Z = trackletBox.transpose()
    # list of sides' polygons of figure
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]],
             [Z[2], Z[3], Z[7], Z[6]]]

    ax.add_collection3d(Poly3DCollection(verts, facecolors=facecolor, linewidths=1, edgecolors=edgecolor, alpha=.25))
    plt.show()
    return ax


plt.close('all')

STATE_UNSET = 0
STATE_INTERP = 1
STATE_LABELED = 2
stateFromText = {'0':STATE_UNSET, '1':STATE_INTERP, '2':STATE_LABELED}

OCC_UNSET = 255  # -1 as uint8
OCC_VISIBLE = 0
OCC_PARTLY = 1
OCC_FULLY = 2
occFromText = {'-1':OCC_UNSET, '0':OCC_VISIBLE, '1':OCC_PARTLY, '2':OCC_FULLY}

TRUNC_UNSET = 255  # -1 as uint8, but in xml files the value '99' is used!
TRUNC_IN_IMAGE = 0
TRUNC_TRUNCATED = 1
TRUNC_OUT_IMAGE = 2
TRUNC_BEHIND_IMAGE = 3
truncFromText = {'99':TRUNC_UNSET, '0':TRUNC_IN_IMAGE, '1':TRUNC_TRUNCATED, \
                  '2':TRUNC_OUT_IMAGE, '3': TRUNC_BEHIND_IMAGE}


kittiDir = '/path/to/kitti/data'
# drive = '2011_09_26_drive_0001'
kittiDir = '/home/becattini/Downloads/2011_09_26/'
drive = '2011_09_26_drive_0048_sync'
xmlParser.example(kittiDir, drive)

twoPi = 2.*np.pi

# read tracklets from file
myTrackletFile = join(kittiDir, drive, 'tracklet_labels.xml')
tracklets = xmlParser.parseXML(myTrackletFile)

# loop over tracklets
for iTracklet, tracklet in enumerate(tracklets):
    print 'tracklet {0: 3d}: {1}'.format(iTracklet, tracklet)

    # this part is inspired by kitti object development kit matlab code: computeBox3D
    h,w,l = tracklet.size
    trackletBox = np.array([ # in velodyne coordinates around zero point and without orientation yet\
        [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
        [ w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
        [ 0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    draw3Dbox(trackletBox)

    all_x = []
    all_y = []
    all_z = []
    # loop over all data in tracklet
    for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:

        # determine if object is in the image; otherwise continue
        if truncation not in (TRUNC_IN_IMAGE, TRUNC_TRUNCATED):
            continue

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]   # other rotations are 0 in all xml files I checked
        assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
        rotMat = np.array([\
            [np.cos(yaw), -np.sin(yaw), 0.0], \
            [np.sin(yaw),  np.cos(yaw), 0.0], \
            [        0.0,          0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8,1)).T

        # calc yaw as seen from the camera (i.e. 0 degree = facing away from cam), as opposed to
        #   car-centered yaw (i.e. 0 degree = same orientation as car).
        #   makes quite a difference for objects in periphery!
        # Result is in [0, 2pi]
        x, y, z = translation
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

        yawVisual = ( yaw - np.arctan2(y, x) ) % twoPi

        im_path = kittiDir + '/' + drive + '/image_02/data/' + str(absoluteFrameNumber).zfill(10) + '.png'
        print im_path
        im = cv2.imread(im_path)
        print im.shape
        # plt.imshow(im)
        # plt.show()
        # cv2.imshow('frame', im)
        # cv2.waitKey(0)
    # ax.scatter3D(all_x, all_y, all_z)
    # plt.show()
