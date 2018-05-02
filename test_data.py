import mxnet as mx
from mxnet.io import ImageRecordIter
from mxnet.image import ImageDetIter
from PIL import Image
import cv2
import numpy as np

data_iter = ImageRecordIter(
    path_imgrec = "/data/deeplearning/dataset/arrow/train_0301.rec",
    path_imglist="/data/deeplearning/dataset/arrow/train_0301.lst",
    label_width=1,
    data_name='data',
    label_name='softmax_label',
    resize=256,
    data_shape=(3, 200, 200),
    batch_size=512,
    pad=0,
    fill_value=127,  # only used when pad is valid
    rand_crop=False,
    max_random_scale=1.0,  # 480 with imagnet and vggface, 384 with msface, 32 with cifar10
    min_random_scale=1.0,  # 256.0/480.0=0.533, 256.0/384.0=0.667
    max_aspect_ratio=0,
    random_h=0,  # 0.4*90
    random_s=0,  # 0.4*127
    random_l=0,  # 0.4*127
    max_rotate_angle=0,
    max_shear_ratio=0,
    rand_mirror=False,
    shuffle=False,
)

data_iter.reset()
next_batch = data_iter.iter_next()
while next_batch:
    data_iter.next()
    print("provide_data:{}".format(data_iter.provide_data[0].shape))
    print("provide_label:{}".format(data_iter.provide_label[0].shape))
    data_index = data_iter.getindex()
    data_label = data_iter.getlabel()
    data = data_iter.getdata()
    print("data:{}".format(data.shape))
    image_data = data[0].asnumpy().astype(np.uint8).transpose((1,2,0))
    cv2.imwrite("./test.jpg", image_data)

    print("label:{}".format(data_label.shape))
    next_batch = data_iter.iter_next()