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

            img = cv2.resize(origin_frame, (224, 224))
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

    class_id_map = {label.id: label.categoryId for label in arrow_labels_v1}
    name_dict = {label.id: label.name for label in arrow_labels_v2}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        print("model file[{}] is not exist".format(model_file))
        exit(0)

    image_dir = args.dir
    dest_dir = os.path.join(args.dir, "../arrow_val")
    #
    # model_net = ModelClassArrow(gpu_id=0)
    # proc_list = []
    # file_list = os.listdir(image_dir)
    # for id_ in file_list:
    #     name_list = str(id_).split(".")
    #     if len(name_list) != 2:
    #         continue
    #
    #     name_only = name_list[0]
    #     name_ext = name_list[1]
    #     if name_ext != 'png' and name_ext != 'jpg':
    #         continue
    #     proc_list.append(id_)
    #
    # if not os.path.exists(dest_dir):
    #     os.makedirs(dest_dir)
    #
    # for id_ in proc_list:
    #     file_path = os.path.join(image_dir, id_)
    #     dest_path = os.path.join(dest_dir, id_)
    #
    #     if os.path.exists(dest_path):
    #         continue
    #
    #     try:
    #         pred_label = None
    #         start = time.time()
    #         with open(file_path, "rb") as f:
    #             img = f.read()
    #             pred_label = model_net.do(image_data=img)
    #         end = time.time()
    #
    #         class_id = pred_label[0]
    #         class_dir = os.path.join(dest_dir, str(class_id))
    #         if not os.path.exists(class_dir):
    #             os.makedirs(class_dir)
    #
    #         dest_path = os.path.join(class_dir, id_)
    #         shutil.copy(file_path, dest_path)
    #         print("Processed {} in {} ms, labels:{}".format(
    #             dest_path, str((end - start) * 1000),
    #             str(pred_label))
    #         )
    #     except Exception as e:
    #         print (repr(e))
    #
    # time_end = time.time()
    # print("finish recognition in {} s\n".format(time_end-time_start))

    print("start to calculate recall and accuracy...\n")
    recall_map = {}
    accuracy_map = {}

    print("loading label...\n")
    val_csv = os.path.join(image_dir, "ImageType.csv")
    if not os.path.exists(val_csv):
        print("label file [{}] is not exist".format(val_csv))
        exit(0)

    train_csv = os.path.join(image_dir, "../train/ImageType.csv")
    train_map = {}
    total_train = 0
    with open(train_csv, "r") as f:
        line_str = f.readline()
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()
            file_name, class_id = line_str.split(",")

            if int(class_id) not in train_map:
                train_map[int(class_id)] = 1
            else:
                train_map[int(class_id)] += 1
            total_train += 1
            line_str = f.readline()
    print(total_train)
    print(train_map)

    label_map = {}
    total_val = 0
    val_map = {}
    with open(val_csv, "r") as f:
        line_str = f.readline()
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()
            file_name, class_id = line_str.split(",")
            label_map[file_name] = class_id

            if int(class_id) not in recall_map:
                recall_map[int(class_id)] = {"total": 1}
            else:
                recall_map[int(class_id)]["total"] += 1

            if int(class_id) not in val_map:
                val_map[int(class_id)] = 1
            else:
                val_map[int(class_id)] += 1
            total_val += 1
            line_str = f.readline()
    print(total_val)
    print(val_map)

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
