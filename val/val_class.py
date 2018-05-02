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
from sign_labels import sign_total_labels

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

        self.input_shape = [112, 112]
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))],
                 label_shapes=[('softmax_label', (1,))])
        self.mod.init_params(arg_params=net_args,
                        aux_params=net_auxs)
        self._flipping = False

    def do(self, image_data):
        pred_data = None
        try:
            image = np.asarray(bytearray(image_data), dtype="uint8")
            origin_frame = cv2.imdecode(image, cv2.COLOR_BGR2RGB)

            # pad image
            img = np.array(Image.fromarray(origin_frame.astype(np.uint8, copy=False)))
            newsize = max(img.shape[:2])
            new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
            margin0 = (newsize - img.shape[0]) // 2
            margin1 = (newsize - img.shape[1]) // 2
            new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
            img = np.resize(new_img, (3, self.input_shape[1], self.input_shape[0]))

            # img = cv2.resize(origin_frame, (224, 224))
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img = img[np.newaxis, :]

            # compute the predict probabilities
            self.mod.forward(Batch([mx.nd.array(img)]))
            prob = self.mod.get_outputs()[0].asnumpy()

            # Return the top-5
            prob = np.squeeze(prob)
            a = np.argsort(prob)[::-1]
            # result = []
            # for i in a[0:5]:
            #     result.append((prob[i].split(" ", 1)[1], round(prob[i], 3)))

            pred_data = a[0:5]

        except Exception as e:
            print("recognition error:{}".format(repr(e)))

        return pred_data

if __name__ == "__main__":
    time_start = time.time()

    name_dict = {}
    for label in sign_total_labels:
        if label.categoryId not in name_dict:
            name_dict[label.categoryId] = label.name

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        print("model file[{}] is not exist".format(model_file))
        exit(0)

    image_dir = args.dir
    dest_dir = os.path.join(args.dir, "../arrow_predict")

    model_net = ModelClassArrow(gpu_id=3)
    proc_list = []

    print("loading test label...\n")
    label_map = {}
    total_val = 0
    recall_map = {}

    dir_list = os.listdir(image_dir)
    for id_dir in dir_list:
        class_dir = os.path.join(image_dir, id_dir)
        file_list = os.listdir(class_dir)
        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue

            proc_list.append(os.path.join(image_dir, id_dir, file_id))
            class_id = id_dir
            label_map[file_id] = class_id
            total_val += 1

            if int(class_id) not in recall_map:
                recall_map[int(class_id)] = {"total": 1}
            else:
                recall_map[int(class_id)]["total"] += 1

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for id_ in proc_list:
        file_path = id_

        try:
            pred_label = None
            start = time.time()
            with open(file_path, "rb") as f:
                img = f.read()
                pred_label = model_net.do(image_data=img)
            end = time.time()

            class_id = pred_label[0]
            class_dir = os.path.join(dest_dir, str(class_id))
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            dest_path = os.path.join(class_dir, os.path.basename(id_))
            shutil.copy(file_path, dest_path)
            print("Processed {} in {} ms, labels:{}".format(
                dest_path, str((end - start) * 1000),
                str(pred_label))
            )
        except Exception as e:
            print (repr(e))

    time_end = time.time()
    print("finish recognition in {} s\n".format(time_end-time_start))

    print("start to calculate recall and accuracy...\n")
    accuracy_map = {}

    print("loading prediction...\n")

    pred_map = {}
    class_dir_list = os.listdir(dest_dir)
    for class_dir in class_dir_list:
        if not str(class_dir).isdigit():
            continue

        pred_dir = os.path.join(dest_dir, class_dir)
        pred_list = os.listdir(pred_dir)

        for pred_file in pred_list:
            if not pred_file.endswith("jpg"):
                continue
            pred_map[pred_file] = class_dir

            if int(class_dir) not in accuracy_map:
                accuracy_map[int(class_dir)] = {"total": 1}
            else:
                accuracy_map[int(class_dir)]["total"] += 1

    correct_map = {}

    for image_name, class_id in label_map.items():
        pred_class = pred_map[image_name]

        if int(class_id) == int(pred_class):
            if int(class_id) not in correct_map:
                correct_map[int(class_id)] = 1
            else:
                correct_map[int(class_id)] += 1

    print("start to calculate recall and accuracy...\n")

    for class_id, count in correct_map.items():
        recall_map[class_id]["correct"] = count
        recall_map[class_id]["rate"] = float(count) / float(recall_map[class_id]["total"]) * 100

        accuracy_map[class_id]["correct"] = count
        accuracy_map[class_id]["rate"] = float(count) / float(accuracy_map[class_id]["total"]) * 100

    for class_id, info in recall_map.items():
        label_name = name_dict[class_id]
        print(label_name.encode("UTF-8"))
        print("recall:id:{},info:{}".format(class_id, json.dumps(info)))

    for class_id, info in accuracy_map.items():
        label_name = name_dict[class_id]
        print(label_name.encode("UTF-8"))
        print("accuracy:id:{},info:{}".format(class_id, json.dumps(info)))

    # format
    for class_id, info in recall_map.items():
        label_name = name_dict[class_id]
        print(label_name.encode("UTF-8"))
        print("recall:id:{},info:{}".format(class_id, json.dumps(info)))

    for class_id, info in accuracy_map.items():
        label_name = name_dict[class_id]
        print(label_name.encode("UTF-8"))
        print("accuracy:id:{},info:{}".format(class_id, json.dumps(info)))
