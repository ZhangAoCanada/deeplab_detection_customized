import os
import numpy as np
import cv2
import pickle
import colorsys
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
            'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush']

def RandomColors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # random.shuffle(colors)
    return colors

# Pre-define all colors
COLORS_ALL = RandomColors(len(CLASS_NAMES))

def ReadMrcnnDict(stereo_mrcnn_dir, frame_i):
    """
    Function:
        Read stereo mask rcnn results w.r.t current frame.

    Args:
        stereo_mrcnn_dir            ->          stereo mask rcnn results directory
        frame_i                     ->          current frame to be read
    """
    pickle_name = stereo_mrcnn_dir + "stereo_mrcnn_" + str(frame_i) + ".pickle"
    if os.path.exists(pickle_name):
        with open(pickle_name, "rb") as f:
            mrcnn_dict = pickle.load(f)
    else:
        mrcnn_dict = None
    return mrcnn_dict

def ReadMrcnnImg(stereo_mrcnn_dir, frame_i):
    """
    Function:
        Read stereo masked images w.r.t current frame

    Args:
        stereo_mrcnn_dir            ->          stereo mask rcnn results directory
        frame_i                     ->          current frame to be read
    """
    img_name = stereo_mrcnn_dir + "stereo_mrcnn_img_" + str(frame_i) + ".jpg"
    if os.path.exists(img_name):
        masked_img = cv2.imread(img_name)
    else:
        masked_img = None
    return masked_img

def ReadStereoOriginalPcl(pcl_dir, frame_i):
    """
    Function:
        Read the stereo point cloud w.r.t current frame
    
    Args:
        stereo_pcl_dir          ->          stereo point cloud directory
        frame_i                 ->          current frame to be read
    """
    original_pcl = np.load(pcl_dir + "initial_" + str(frame_i) + ".npy")
    thresholed_pcl = np.load(pcl_dir + "organized_" + str(frame_i) + ".npy")
    return original_pcl, thresholed_pcl

def ObjectPcl(original_pcl, mask):
    """
    Function:
        apply mask onto the original pcl to find the object.
    """
    mask = np.expand_dims(mask, -1)
    object_pcl = np.where(mask == 1, original_pcl, 0)
    return object_pcl

def TransferToScatterPoints(points_3d):
    """
    Function:
        Transfer the 3d matrices pcl to 2d matrices pcl

    Args:
        points_3d           ->          target 3d points
    """
    non_zero_points = points_3d.copy()
    non_zero_mask = non_zero_points[..., -1] > 0
    non_zero_points = non_zero_points[non_zero_mask]
    return non_zero_points

def main(stereo_mrcnn_dir, stereo_pcl_dir, save_dir, sequence, if_plot=True):
    """
    MAIN FUNCTION

    ATTENTION:
        mrcnn_dict['rois']          ->          bounding boxes
        mrcnn_dict['masks']         ->          masks
        mrcnn_dict['class_ids']      ->          class ID (look up in the CLASS_NAMES)
        mrcnn_dict['scores']        ->          scores
    """ 
    designated_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'train',  
                    'truck', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter']

    ##### for plotting #####
    if if_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    for frame_id in tqdm(range(sequence[0], sequence[1])):
        mrcnn_dict = ReadMrcnnDict(stereo_mrcnn_dir, frame_id)
        mrcnn_img = ReadMrcnnImg(stereo_mrcnn_dir, frame_id)
        if mrcnn_dict is None or len(mrcnn_dict['rois']) == 0:
            print("frame %d has no mrcnn prediction."%frame_id)
            continue
        ##### for plotting #####
        if if_plot:
            plt.cla()
            ax1.clear()
            ax1.imshow(mrcnn_img[..., ::-1])
            ax2.clear()

        stereo_obj_pcl_dict = {'class': [], 'pcl':[]}
        stereo_origin_pcl, _ = ReadStereoOriginalPcl(stereo_pcl_dir, frame_id)
        for object_i in range(len(mrcnn_dict['rois'])):
            current_class = CLASS_NAMES[mrcnn_dict['class_ids'][object_i]]
            
            if current_class in designated_classes:
                object_pcl = ObjectPcl(stereo_origin_pcl, \
                            mrcnn_dict['masks'][..., object_i].astype(np.int))
                y1, x1, y2, x2 = mrcnn_dict['rois'][object_i]
                scores = mrcnn_dict['scores'][object_i]
                object_pcl_3d = TransferToScatterPoints(object_pcl)
                stereo_obj_pcl_dict['class'].append(current_class)
                stereo_obj_pcl_dict['pcl'].append(object_pcl_3d)
                ##### for plotting ######
                if if_plot:    
                    ax2.scatter(object_pcl_3d[:, 0], object_pcl_3d[:,2], s=0.5, \
                                c=COLORS_ALL[designated_classes.index(current_class)], \
                                label=current_class)
        with open(save_dir + "stereo_obj_pcl_" + str(frame_id) + ".pickle", "wb") as f:
            pickle.dump(stereo_obj_pcl_dict, f)
        ##### for plotting #####
        if if_plot:
            ax2.set_xlim([-15, 15])
            ax2.set_ylim([0, 25])
            ax2.legend()
            fig.canvas.draw()
            plt.pause(0.1)
 
if __name__ == "__main__":
    stereo_mrcnn_dir = "./stereo_object_detection_output/"
    stereo_pcl_dir = "../stereo_process/pcl/"
    save_dir = "./stereo_object_pcl/" 
    sequence = [1000, 1500]
    main(stereo_mrcnn_dir, stereo_pcl_dir, save_dir, sequence, if_plot=False)


