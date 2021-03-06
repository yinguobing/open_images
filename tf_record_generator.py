import tensorflow as tf
import os
import numpy as np
import cv2
from tqdm import tqdm


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """Returns a float_list from a float / double list."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint list."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


class DetectionSample(object):

    def __init__(self, image_file, boxes):
        """Construct an object detection sample.

        Args:
            image: image file path.
            boxes: numpy array of bounding boxes [[x_min, y_min, x_max, y_max], ...]

        """
        self.image_file = image_file
        self.boxes = boxes

    def read_image(self, format="BGR"):
        """Read in image as numpy array in format of BGR by defult, else RGB.

        Args:
            format: the channel order, default BGR.

        Returns:
            a numpy array.
        """
        img = cv2.imread(self.image_file)
        if format != "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img


def create_tf_example(example, min_size=None, max_samples=100):

    img = example.read_image()
    height, width, _ = img.shape

    # Filename of the image.
    filename = example.image_file.split('/')[-1].encode("utf-8")

    # Encoded image bytes
    with tf.io.gfile.GFile(example.image_file, 'rb') as fid:
        encoded_image_data = fid.read()

    image_format = example.image_file.split('.')[-1].encode("utf-8")

    # Transform the bbox size.
    boxes_oid6 = example.boxes
    xmin, ymin, xmax, ymax = np.split(boxes_oid6, 4, axis=1)

    xmin *= width
    xmax *= width
    ymin *= height
    ymax *= height

    wbox = xmax - xmin
    hbox = ymax - ymin

    # Filter boxes whose size exceeds the threshold.
    if min_size:
        mask = np.all((wbox > min_size, hbox > min_size))

        # Incase all boxes are invalid.
        if not np.any(mask):
            print("No valid face box found.")
            return None

        xmin = xmin[mask]
        ymin = ymin[mask]
        wbox = wbox[mask]
        hbox = hbox[mask]
    
    # Incase too many boxes.
    if xmin.shape[0] > max_samples:
        print("Too many boxes: {}.".format(xmin.shape[0]))
        return None

    # In case the coordinates are flipped.
    xs = np.concatenate((xmin, xmin+wbox), axis=-1)
    ys = np.concatenate((ymin, ymin+hbox), axis=-1)

    xmin = np.min(xs, axis=-1)
    xmax = np.max(xs, axis=-1)
    ymin = np.min(ys, axis=-1)
    ymax = np.max(ys, axis=-1)

    # Make sure all boxes are in image boundaries.
    ymax = (np.clip(ymax, a_min=0, a_max=height).flatten() / height).tolist()
    xmax = (np.clip(xmax, a_min=0, a_max=width).flatten() / width).tolist()
    ymin = (np.clip(ymin, a_min=0, a_max=height).flatten() / height).tolist()
    xmin = (np.clip(xmin, a_min=0, a_max=width).flatten() / width).tolist()

    # List of integer class id of bounding box (1 per box)
    classes = [1 for _ in xmin]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/encoded': bytes_feature(encoded_image_data),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/ymin': float_list_feature(ymin),
        'image/object/bbox/xmin': float_list_feature(xmin),
        'image/object/bbox/ymax': float_list_feature(ymax),
        'image/object/bbox/xmax': float_list_feature(xmax),
        'image/object/class/label': int64_list_feature(classes)
    }))

    return tf_example


if __name__ == "__main__":
    writer = tf.io.TFRecordWriter(
        "/home/robin/data/face/wider/tfrecord/wider_train.record")

    # Read in your dataset to examples variable
    data_dir = "/home/robin/data/face/wider"
    wider = WiderFace(data_dir, mode="train")

    for example in tqdm(wider):
        tf_example = create_tf_example(example, min_size=64)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())

    writer.close()
