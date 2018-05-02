# -*-coding:utf-8-*-
import sys
import random

sys.path.insert(0, "/opt/densenet.mxnet")

import os
import time
import argparse
import shutil

from sign_labels import sign_total_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--check_dir', type=str, required=True)
    args = parser.parse_args()

    label_map = {}
    label_name = {}
    for label in sign_total_labels:
        label_map[label.label] = label.categoryId
        label_name[label.label] = label.name

    label_dir = args.label_dir
    check_dir = args.check_dir

    if not os.path.exists(label_dir):
        print("dir[{}] is not exist".format(label_dir))
        exit(0)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    time_start = time.time()

    class_file = {}
    sort_id = []
    dir_list = os.listdir(label_dir)
    for dir_name in dir_list:
        dir_path = os.path.join(label_dir, dir_name)
        if os.path.isdir(dir_path):
            file_list = os.listdir(dir_path)
            train_id = dir_name
            sort_id.append(train_id)
            if train_id not in class_file:
                class_file[train_id] = []
            for file_id in file_list:
                if not file_id.endswith("jpg"):
                    continue
                class_file[train_id].append(os.path.join(dir_path, file_id))
        else:
            continue
    sort_id.sort()

    for train_id in sort_id:
        file_list = class_file[train_id]
        class_name = label_name[train_id]
        chose = len(file_list)
        if chose > 100:
            chose = 100
        print("{},{},{},{}".format(
            train_id, chose, len(file_list), class_name.encode("UTF-8")
        ))

    chose_count = 100
    for train_id, file_list in class_file.items():
        cur_len = len(file_list)
        if cur_len <= 100:
            chose_count = cur_len
        else:
            chose_count = 100

        if len(file_list) > chose_count:
            # random.seed(random.randint(10, 100))
            random.shuffle(file_list)
            chose_list = file_list[:chose_count]
        else:
            chose_list = file_list

        id_dir = os.path.join(check_dir, str(train_id))
        if not os.path.exists(id_dir):
            os.makedirs(id_dir)

        for file_id in chose_list:
            shutil.copy(src=os.path.join(label_dir, file_id),
                        dst=os.path.join(id_dir, os.path.basename(file_id)))

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))