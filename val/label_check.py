# -*-coding:utf-8-*-

import sys
sys.path.insert(0, "/opt/densenet.mxnet")

import numpy as np
import os
import json
import mxnet as mx
import time
import argparse
import cv2
from PIL import Image
from collections import namedtuple
import shutil

from util import load_weights
from preprocess.class_map import arrow_labels_v1
from preprocess.class_map import arrow_labels_v2

# define a simple data batch
Batch = namedtuple('Batch', ['data'])


class ModelClassArrow:
    def __init__(self, gpu_id=0):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)

        # model_file_name = "densenet-kd-169-0-5000.params"
        self.weights = model_file

        network, net_args, net_auxs = load_weights(self.weights)
        context = [mx.gpu(gpu_id)]
        self.mod = mx.mod.Module(network, context=context)

        self.input_shape = [256, 256] # (W, H)
        # self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.input_shape[1], self.input_shape[0]))],
        #               label_shapes=[('softmax_label', (1,))])
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.input_shape[1], self.input_shape[0]))],
                      label_shapes=None)
        self.mod.init_params(arg_params=net_args,
                        aux_params=net_auxs)
        self._flipping = False

    def do(self, image_data):
        pred_data = None
        accuracy = 0
        try:
            image = np.asarray(bytearray(image_data), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            print "original img size:", img.shape

            # pad image
            # img = np.array(Image.fromarray(origin_frame.astype(np.uint8, copy=False)))
            newsize = max(img.shape[:2])
            new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
            margin0 = (newsize - img.shape[0]) // 2
            margin1 = (newsize - img.shape[1]) // 2
            new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
            # img: (256, 256, 3), GBR format, HWC
            img = cv2.resize(new_img, tuple(self.input_shape))
            print "resized img size:", img.shape

            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :]

            # compute the predict probabilities
            self.mod.forward(Batch([mx.nd.array(img)]))
            prob = self.mod.get_outputs()[0].asnumpy()

            # Return the top-5
            prob = np.squeeze(prob)
            acc = np.sort(prob)[::-1]
            a = np.argsort(prob)[::-1]
            # result = []
            # for i in a[0:5]:
            #     result.append((prob[i].split(" ", 1)[1], round(prob[i], 3)))

            pred_data = a[0:5]
            accuracy = acc[0:5]

        except Exception as e:
            print("recognition error:{}".format(repr(e)))

        return pred_data, accuracy

if __name__ == "__main__":
    time_start = time.time()

    class_id_map = {label.id: label.categoryId for label in arrow_labels_v1}
    name_dict = {label.id: label.name for label in arrow_labels_v2}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--augment', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--result', type=str, required=True)
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        print("model file[{}] is not exist".format(model_file))
        exit(0)

    image_dir = args.dir
    augments = args.augment
    dest_dir = args.result

    model_net = ModelClassArrow(gpu_id=3)
    proc_list = []

    label_map = {}
    recall_map = {}
    dir_list = os.listdir(image_dir)
    for id_dir in dir_list:
        if not id_dir.isdigit():
            continue

        class_dir = os.path.join(image_dir, id_dir)
        file_list = os.listdir(class_dir)
        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue

            proc_list.append(os.path.join(image_dir, id_dir, file_id))
            class_id = id_dir
            label_map[file_id] = class_id

            if int(class_id) not in recall_map:
                recall_map[int(class_id)] = {"total": 1}
            else:
                recall_map[int(class_id)]["total"] += 1

    dir_list = os.listdir(augments)
    for id_dir in dir_list:
        if not id_dir.isdigit():
            continue

        class_dir = os.path.join(augments, id_dir)
        file_list = os.listdir(class_dir)
        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue

            proc_list.append(os.path.join(augments, id_dir, file_id))
            class_id = id_dir
            label_map[file_id] = class_id

            if int(class_id) not in recall_map:
                recall_map[int(class_id)] = {"total": 1}
            else:
                recall_map[int(class_id)]["total"] += 1

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    result_file = os.path.join(dest_dir, "info.csv")
    with open(result_file, "w") as f1:
        f1.write("file,label,predict,score\n")
        for id_ in proc_list:
            file_path = id_

            try:
                pred_label = None
                start = time.time()
                assert os.path.exists(file_path)
                with open(file_path, "rb") as f:
                    img = f.read()
                    pred_label, accuracy = model_net.do(image_data=img)
                end = time.time()

                class_id = str(pred_label[0])
                class_acc = accuracy[0]

                label_id = str(os.path.basename(os.path.dirname(id_)))
                if class_acc < 0.8:
                    if class_id == label_id:
                        _dest_dir = os.path.join(dest_dir, "same.low")
                    else:
                        _dest_dir = os.path.join(dest_dir, "diff.low")
                else:
                    if class_id == label_id:
                        _dest_dir = os.path.join(dest_dir, "same.high")
                    else:
                        _dest_dir = os.path.join(dest_dir, "diff.high")

                class_dir = os.path.join(_dest_dir, str(class_id))
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                dest_path = os.path.join(class_dir, os.path.basename(id_))
                shutil.copy(file_path, dest_path)
                print("Processed {} in {} ms,\nacc:{}, labels:{} vs. {}".format(
                    os.path.basename(dest_path), str((end - start) * 1000),
                    class_acc,
                    str(pred_label[0]), label_map[os.path.basename(file_path)])
                )
                msg = "{},{},{},{}\n".format(os.path.basename(id_), label_id, class_id, class_acc)
                f1.write(msg)
            except Exception as e:
                print (repr(e))

    time_end = time.time()
    print("finish recognition in {} s\n".format(time_end-time_start))
