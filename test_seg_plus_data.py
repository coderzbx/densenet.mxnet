# -*-coding:utf-8-*-

from mxnet.io import ImageSegRecordIter
import os
import cv2
import time
import numpy as np

from collections import namedtuple

Label = namedtuple(
    'Label', ['en_name', 'id', 'categoryId', 'color', 'name'])
#
# self_road_chn_labels = {
#     Label('other',          0, 0,     (64, 64, 32),       u'其他'),
#     Label('ignore',         1, 1,     (0, 0, 0),          u'Ignore'),
#     Label('lane',           2, 2,     (255, 0, 0),        u'车道标线-白色'),
#     Label('left',           3, 3,     (255, 192, 203),    u'左侧道路边缘线'),
#     Label('right',          4, 4,     (139, 0, 139),      u'右侧道路边缘线'),
#     Label('v_slow',         5, 5,     (32, 128, 192),     u'纵向减速标线'),
#     Label('bus_lane',       6, 6,     (192, 128, 255),    u'专用车道标线'),
#     Label('stop',           7, 7,     (255, 128, 64),     u'停止线'),
#     Label('slow_let',       8, 8,     (0, 255, 255),      u'减速让行标线'),
#     Label('slow_zone',      9, 9,     (128, 128, 255),    u'减速标线/减速带'),
#     Label('sidewalk',       10, 10,   (128, 192, 192),    u'人行横道'),
#     Label('connection',     11, 11,   (128, 128, 192),    u'路面连接带'),
#     Label('stop_station',   12, 12,   (240, 128, 128),    u'停靠站标线'),
#     Label('in_out',         13, 13,   (128, 128, 0),      u'出入口标线'),
#     Label('symbol',         14, 14,   (0, 0, 255),        u'文字符号类'),
#     Label('fish_lane',      15, 15,   (0, 255, 0),        u'导流线（鱼刺线）'),
#     Label('stop_gird',      16, 16,   (255, 255, 0),      u'停止网格标线'),
#     Label('distance',       17, 17,   (255, 128, 255),    u'车距确认线'),
#     Label('road',           18, 18,   (192, 192, 192),    u'道路'),
#     Label('objects',        19, 19,   (128, 0, 0),        u'车辆及路面上其他物体'),
#     Label('curb',           20, 20,   (0, 139, 139),      u'虚拟车道线-路缘石'),
#     Label('fence',          21, 21,   (255, 106, 106),    u'虚拟车道线-防护栏'),
#     Label('objects',        22, 22,   (118, 180, 254),    u'虚拟车道线-其他'),
#     Label('curb',           23, 23,   (144, 238, 144),    u'左弯待转区线'),
#     Label('fence',          24, 25,   (0, 255, 127),      u'可变导向车道线'),
#     Label('objects',        25, 25,   (255, 165, 0),      u'车道标线-黄色'),
# }

self_road_chn_labels = {
    Label('other',          0, 0,     (64, 64, 32),       u'其他'),
    Label('ignore',         1, 1,     (0, 0, 0),          u'Ignore'),
    Label('lane',           2, 2,     (255, 0, 0),        u'车道标线-白色'),
    Label('left',           3, 3,     (255, 192, 203),    u'左侧道路边缘线'),
    Label('right',          4, 4,     (139, 0, 139),      u'右侧道路边缘线'),
    Label('v_slow',         5, 2,     (32, 128, 192),     u'纵向减速标线'),
    Label('bus_lane',       6, 2,     (192, 128, 255),    u'专用车道标线'),
    Label('stop',           7, 2,     (255, 128, 64),     u'停止线'),
    Label('slow_let',       8, 2,     (0, 255, 255),      u'减速让行标线'),
    Label('slow_zone',      9, 2,     (128, 128, 255),    u'减速标线/减速带'),
    Label('sidewalk',       10, 3,   (128, 192, 192),    u'人行横道'),
    Label('connection',     11, 3,   (128, 128, 192),    u'路面连接带'),
    Label('stop_station',   12, 5,   (240, 128, 128),    u'停靠站标线'),
    Label('in_out',         13, 5,   (128, 128, 0),      u'出入口标线'),
    Label('symbol',         14, 6,   (0, 0, 255),        u'文字符号类'),
    Label('fish_lane',      15, 6,   (0, 255, 0),        u'导流线（鱼刺线）'),
    Label('stop_gird',      16, 2,   (255, 255, 0),      u'停止网格标线'),
    Label('distance',       17, 3,   (255, 128, 255),    u'车距确认线'),
    Label('road',           18, 3,   (192, 192, 192),    u'道路'),
    Label('objects',        19, 2,   (128, 0, 0),        u'车辆及路面上其他物体'),
    Label('curb',           20, 4,   (0, 139, 139),      u'虚拟车道线-路缘石'),
    Label('fence',          21, 4,   (255, 106, 106),    u'虚拟车道线-防护栏'),
    Label('objects',        22, 5,   (118, 180, 254),    u'虚拟车道线-其他'),
    Label('curb',           23, 5,   (144, 238, 144),    u'左弯待转区线'),
    Label('fence',          24, 6,   (0, 255, 127),      u'可变导向车道线'),
    Label('objects',        25, 6,   (255, 165, 0),      u'车道标线-黄色'),
}

with open("/opt/densenet.mxnet/label_map.txt", "w") as f:
    label_ids = []
    for label in self_road_chn_labels:
        label_ids.append(label.id)

    label_ids.sort()
    for label_id in label_ids:
        for label in self_road_chn_labels:
            if label_id == label.id:
                line_str = "{}\t{}\t{}\n".format(label.id, label.categoryId, label.en_name)
                f.write(line_str)
                break

batch_size = 64
data_iter = ImageSegRecordIter(
    path_imgrec = "/data/deeplearning/dataset/training/seg_plus_data.rec",
    # path_imglist="/data/deeplearning/dataset/training/seg_data.lst",
    # label_width=1,
    data_name='data',
    label_name='softmax_label',
    data_shape=(3, 400, 400),
    batch_size=batch_size,
    preprocess_threads=batch_size,
    # resize=512,
    pad=0,
    fill_value=255,  # only used when pad is valid
    rand_crop=True,
    max_random_scale=0.7,
    min_random_scale=0.5,
    max_aspect_ratio=0.4,
    # max_random_scale=0.65,
    # min_random_scale=0.4,
    # max_aspect_ratio=0.45,
    # max_random_scale=1.0,
    # min_random_scale=1.0,
    # max_aspect_ratio=0,
    random_h=0,
    random_s=0,
    random_l=0,
    max_rotate_angle=0,
    max_shear_ratio=0,
    rand_mirror=True,
    rand_mirror_prob=0.4,
    left_lane_id=3,
    right_lane_id=4,
    shuffle=False,
    label_map_file="/opt/densenet.mxnet/label_map.txt",
    label_scale=0.3
)

data_iter.reset()
next_batch = data_iter.iter_next()
image_index = 0
batch_index = 0
save_image = False

if not os.path.exists("/opt/densenet.mxnet/test_plus_batch"):
    os.makedirs("/opt/densenet.mxnet/test_plus_batch")

while next_batch:
    try:
        data_iter.next()
    except StopIteration:
        data_iter.reset()
        next_batch = data_iter.iter_next()
        data_iter.next()

    batch_start = time.time()

    i = 0
    data_index = data_iter.getindex()
    data_label = data_iter.getlabel()
    data = data_iter.getdata()

    print("label:{}".format(data_label.shape))
    print("data:{}".format(data.shape))

    if batch_index == 0 and save_image:
        for i in range(batch_size):
            image_index = i
            start = time.time()
            print("provide_data:{}".format(data_iter.provide_data[i].shape))
            print("provide_label:{}".format(data_iter.provide_label[i].shape))
            _label = data_label[i].asnumpy().astype(np.uint8)
            # cv2.imwrite("/opt/densenet.mxnet/test_plus_batch/{}.png".format(image_index), _label)
            # label
            height = _label.shape[0]
            width = _label.shape[1]
            blank_image = np.zeros((height, width, 3), np.uint8)
            for label in self_road_chn_labels:
                color = label.color
                color = (color[2], color[1], color[0])
                # color = color[::-1]
                blank_image[np.where((_label == label.id))] = color
            cv2.imwrite("/opt/densenet.mxnet/test_plus_batch/{}-{}-label.png".format(batch_index, image_index), blank_image)

            image_label = data[i].asnumpy().astype(np.uint8).transpose(1, 2, 0)
            # image = image_label[:, :, 0:3]
            image = image_label[:, :, ::-1]
            cv2.imwrite("/opt/densenet.mxnet/test_plus_batch/{}-{}.jpg".format(batch_index, image_index), image)

            end = time.time()
            # print("batch{}-{} in {}s".format(batch_index, image_index, (end - start)))

    next_batch = data_iter.iter_next()

    batch_end = time.time()
    print("batch-{} time:{} s".format(batch_index, batch_end-batch_start))

    batch_index += 1
    # if batch_index == 1:
    #     break
    if not next_batch:
        data_iter.reset()
        next_batch = data_iter.iter_next()
    if batch_index == 10:
        break
exit(0)
