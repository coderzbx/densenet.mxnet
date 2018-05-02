import mxnet as mx
from mxnet.io import ImageRecordIter
from PIL import Image
import cv2
import numpy as np
import sys
import os

data_dir = "./test_batch"
rec_file = "/data/deeplearning/dataset/sign/record/20180427_val.rec"
record = mx.recordio.MXRecordIO(rec_file, 'r')

data_iter = ImageRecordIter(
    path_imgrec = rec_file,
    label_width=1,
    data_name='data',
    label_name='softmax_label',
    resize=128,
    data_shape=(3, 112, 112),
    batch_size=1,
    # max_img_size=128,
    # min_img_size=128,
    pad=0,
    fill_value=127,  # only used when pad is valid
    rand_crop=True,
    # max_random_scale=0.8,
    # min_random_scale=0.5,
    # max_aspect_ratio=0.667,
    # min_aspect_ratio=0.375,
    random_h=0,
    random_s=0,
    random_l=0,
    max_rotate_angle=0,
    max_shear_ratio=0,
    rand_mirror=False,
    shuffle=False,
)

data_iter.reset()
n = 0
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

while True:
    data_iter.next()
    data_index = data_iter.getindex()
    data_label = data_iter.getlabel()
    data = data_iter.getdata()
    #print("data:{}".format(data.shape))
    image_data = data[0].asnumpy().astype(np.uint8).transpose((1, 2, 0))
    image_data = image_data[:, :, ::-1]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    label_id = int(data_label.asnumpy()[0])
    class_dir = os.path.join(data_dir, str(label_id))
    # if not os.path.exists(class_dir):
    #     os.makedirs(class_dir)
    file_name = "{}/test_ori{}.jpg".format(data_dir, n)
    file_path = os.path.join(class_dir, file_name)
    cv2.imwrite(file_name, image_data)

    # print("label:{}".format(data_label.shape))
    print n, data_label.asnumpy()[0]

    if n > 100:
       break
    n += 1

