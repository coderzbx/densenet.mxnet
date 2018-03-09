# -*-coding:utf-8-*-
import sys
import random

sys.path.insert(0, "/opt/densenet.mxnet")

import os
import time
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_dir', type=str, required=True)
    parser.add_argument('--rotate_dir', type=str, required=True)
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    args = parser.parse_args()

    label_dir = args.label_dir
    rotate_dir = args.rotate_dir
    train_dir = args.train_dir
    val_dir = args.val_dir

    if not os.path.exists(label_dir):
        print("dir[{}] is not exist".format(label_dir))
        exit(0)
    if not os.path.exists(rotate_dir):
        print("dir[{}] is not exist".format(rotate_dir))
        exit(0)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    time_start = time.time()

    class_file = {}
    class_dir_list = os.listdir(label_dir)
    total_list = []
    for class_id in class_dir_list:
        class_dir = os.path.join(label_dir, class_id)
        file_list = os.listdir(class_dir)

        for file_id in file_list:
            if len(file_id) < 4 or file_id[-3:] not in ['jpg', 'png']:
                file_list.remove(file_id)
                continue

        result_list = []
        if len(file_list) >= 1000:
            random.seed(random.randint(0, 100))
            random.shuffle(file_list)
            result_list.extend([os.path.join(class_dir, file_id) for file_id in file_list[:]])
        else:
            # label
            result_list.extend([os.path.join(class_dir, file_id) for file_id in file_list])

            # rotate
            rotate_class_dir = os.path.join(rotate_dir, class_id)
            rotate_list = os.listdir(rotate_class_dir)
            for rotate_id in rotate_list:
                if len(rotate_id) < 4 or rotate_id[-3:] not in ['jpg', 'png']:
                    rotate_list.remove(rotate_id)
                    continue
            result_list.extend([os.path.join(rotate_class_dir, rotate_id) for rotate_id in rotate_list])

        print("train-class:{}, count:{}".format(class_id, len(result_list)))

        for result_id in result_list:
            train_class_dir = os.path.join(train_dir, class_id)
            if not os.path.exists(train_class_dir):
                os.makedirs(train_class_dir)

            result_path = os.path.join(train_class_dir, os.path.basename(result_id))
            shutil.copy(result_id, result_path)

        total_list.extend(result_list[:])

    random.seed(random.randint(0, 100))
    random.shuffle(total_list)

    val_list = total_list[0:5000]
    for result_id in val_list:
        class_id = os.path.dirname(os.path.dirname(result_id))
        train_class_dir = os.path.join(train_dir, class_id)
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)

        result_path = os.path.join(train_class_dir, os.path.basename(result_id))
        shutil.copy(result_id, result_path)

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))