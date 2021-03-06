import mxnet as mx
from mxnet.io import ImageRecordIter
from PIL import Image
import cv2
import numpy as np
import sys
import os

data_dir = "./val_data"
rec_file = "/data/deeplearning/dataset/arrow/record/data_0319_val.rec"
record = mx.recordio.MXRecordIO(rec_file, 'r')

#n = 0
#while True:
#    item = record.read()
#    header, s = mx.recordio.unpack(item)
#    print header
#    with open("test_img{}.jpg".format(n), "wb") as f:
#        f.write(s)
#    n += 1
#    if n > 30:
#        break
#
#sys.exit(0)

data_iter = ImageRecordIter(
    path_imgrec = rec_file,
    #path_imglist="/data/deeplearning/dataset/arrow/train_0301.lst",
    label_width=1,
    data_name='data',
    label_name='softmax_label',
    resize=256,
    data_shape=(3, 224, 224),
    batch_size=1,
    max_img_size=256,
    min_img_size=256,
    pad=0,
    fill_value=127,  # only used when pad is valid
    rand_crop=True,
    max_random_scale=1.1,  # 480 with imagnet and vggface, 384 with msface, 32 with cifar10
    min_random_scale=0.9,  # 256.0/480.0=0.533, 256.0/384.0=0.667
    max_aspect_ratio=0.1,
    random_h=0,  # 0.4*90
    random_s=0,  # 0.4*127
    random_l=0,  # 0.4*127
    max_rotate_angle=5,
    max_shear_ratio=0,
#    max_random_contrast=,
#    max_random_illumination=,
    rand_mirror=False,
    shuffle=False,
)

#data_iter = ImageRecordIter(
#    path_imgrec = "/data/deeplearning/dataset/arrow/train_0301.rec",
#    #path_imglist="/data/deeplearning/dataset/arrow/train_0301.lst",
#    label_width=1,
#    data_name='data',
#    label_name='softmax_label',
#    resize=256,
#    data_shape=(3, 112, 112),
#    batch_size=1,
#    max_img_size=256,
#    min_img_size=256,
#    pad=0,
#    fill_value=127,  # only used when pad is valid
#    rand_crop=False,
#    max_random_scale=1.0,  # 480 with imagnet and vggface, 384 with msface, 32 with cifar10
#    min_random_scale=1.0,  # 256.0/480.0=0.533, 256.0/384.0=0.667
#    max_aspect_ratio=0.0,
#    random_h=0,  # 0.4*90
#    random_s=0,  # 0.4*127
#    random_l=0,  # 0.4*127
#    max_rotate_angle=0,
#    max_shear_ratio=0,
##    max_random_contrast=,
##    max_random_illumination=,
#    rand_mirror=False,
#    shuffle=False,
#)

data_iter.reset()
n = 0
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
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    file_name = "test_ori{}.jpg".format(n)
    file_path = os.path.join(class_dir, file_name)
    cv2.imwrite(file_path, image_data)

    # print("label:{}".format(data_label.shape))
    print n, data_label.asnumpy()[0]

    # if n > 30:
    #    break
    n += 1

