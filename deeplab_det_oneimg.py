import os
import tarfile
import colorsys
import random
from six.moves import urllib
import cv2
import numpy as np
import tensorflow as tf

from tqdm import tqdm

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break

        tar_file.close()

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        # width, height = image.size
        # resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        # batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                # feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        resized_image = []
        for i in range(len(image)):
            current_img = image[i]
            width, height = current_img.shape[1], current_img.shape[0]
            resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
            target_size = (int(resize_ratio * width), int(resize_ratio * height))
            resized_img = cv2.resize(current_img, target_size)
            resized_image.append(resized_img)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: resized_image})
        return resized_image, batch_seg_map


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(11011)
    random.shuffle(colors)
    return colors

def detectImages(MODEL, imgs):
    resized_imgs, seg_maps = MODEL.run(imgs)
    return seg_maps

def maskImage(images, masks, idxes, CLASS_NAMES, ALL_COLORS, save_dir, save_prefix):
    """Function: mask all images with original resolution"""
    num_imgs = len(images)
    for i in range(num_imgs):
        img = images[i]
        frame_ind = idxes[i]
        mask = masks[i]
        all_classes = np.unique(mask)
        for j in range(len(all_classes)):
            class_ind = all_classes[j]
            if class_ind == 0:
                continue
            mask_class = np.where(mask == class_ind, 1., 0.)
            mask_resize = np.round(cv2.resize(mask_class, (img.shape[1], img.shape[0]), \
                                            cv2.INTER_NEAREST))
            mask_resize = np.where(mask_resize > 0, 1., 0.)
            class_name = CLASS_NAMES[class_ind]
            color = ALL_COLORS[class_ind]
            apply_mask(img, mask_resize, color, alpha=0.5)
        save_name = os.path.join(save_dir, save_prefix + "%.d.png"%(frame_ind))
        cv2.imwrite(save_name, img[..., ::-1])

def inputGenerator(input_dir, input_prefix, sequences, batch_size):
    """Generating input images w.r.t batch size"""
    input_imgs = []
    input_frame_ids = []
    for frame_idx in tqdm(range(sequences[0], sequences[1])):
        img_name = os.path.join(input_dir, input_prefix + "%.d.jpg"%(frame_idx))
        img = cv2.imread(img_name)
        if img is None:
            continue
        img = img[..., ::-1]
        input_imgs.append(img)
        input_frame_ids.append(frame_idx)
        if len(input_imgs) == batch_size or frame_idx == sequences[1]:
            yield input_imgs, input_frame_ids
            input_imgs = []
            input_frame_ids = []

def main(input_dir, input_prefix, sequences):
    save_dir = "./output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_prefix = "dplb_"
    batch_size = 1

    CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', \
                    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', \
                    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', \
                    'train', 'tv']

    MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

    _DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
    _MODEL_URLS = {
        'mobilenetv2_coco_voctrainaug':
            'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
        'mobilenetv2_coco_voctrainval':
            'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
        'xception_coco_voctrainaug':
            'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
        'xception_coco_voctrainval':
            'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
    }
    _TARBALL_NAME = MODEL_NAME + '.tar.gz'

    model_dir = "./deeplab_models_download"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    download_path = os.path.join(model_dir, _TARBALL_NAME)

    if not os.path.exists(download_path):
        print('downloading model, this might take a while...')
        urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME],
                           download_path)
        print('download completed! loading DeepLab model...')

    MODEL = DeepLabModel(download_path)
    print('model loaded successfully!')
    ALL_COLORS = random_colors(len(CLASS_NAMES))

    print("----------------------------------------------------")
    print("---------------- start detection -------------------")
    print("----------------------------------------------------")
    for batch_imgs, batch_idx in inputGenerator(input_dir, input_prefix, \
                                                sequences, batch_size):
        batch_masks = detectImages(MODEL, batch_imgs)
        maskImage(batch_imgs, batch_masks, batch_idx, CLASS_NAMES, ALL_COLORS,\
                    save_dir, save_prefix)

if __name__ == "__main__":
    input_dir = "./stereo_input"
    input_prefix = "stereo_input_"
    sequences = [0, 750]
    main(input_dir, input_prefix, sequences)
