# -*-coding:utf-8-*-

import sys
sys.path.insert(0, "/opt/densenet.mxnet")

import numpy as np
import os
import mxnet as mx
import time
import argparse
import cv2
from collections import namedtuple
import shutil

from util import load_weights
from preprocess.class_map import arrow_labels_v1

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()

    model_file = args.model
    if not os.path.exists(model_file):
        print("model file[{}] is not exist".format(model_file))
        exit(0)

    model_net = ModelClassArrow(gpu_id=0)

    package_list = os.listdir(args.dir)
    for package_dir in package_list:
        if not package_dir.isdigit():
            continue

        image_dir = os.path.join(args.dir, package_dir)

        proc_list = []
        file_list = os.listdir(image_dir)
        for id_ in file_list:
            name_list = str(id_).split(".")
            if len(name_list) != 2:
                continue

            name_only = name_list[0]
            name_ext = name_list[1]
            if name_ext != 'png' and name_ext != 'jpg':
                continue
            proc_list.append(id_)

        dest_dir = os.path.join(args.dir, "arrow_recognition", package_dir)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for id_ in proc_list:
            file_path = os.path.join(image_dir, id_)
            dest_path = os.path.join(dest_dir, id_)

            if os.path.exists(dest_path):
                continue

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

                dest_path = os.path.join(class_dir, id_)
                shutil.copy(file_path, dest_path)
                print("Processed {} in {} ms, labels:{}".format(
                    dest_path, str((end - start) * 1000),
                    str(pred_label))
                )
            except Exception as e:
                print (repr(e))
    time_end = time.time()
    print("finish in {} s".format(time_end-time_start))