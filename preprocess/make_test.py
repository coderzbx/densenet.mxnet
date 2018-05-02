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
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    args = parser.parse_args()

    val_dir = args.val_dir
    test_dir = args.test_dir

    if not os.path.exists(val_dir):
        print("dir[{}] is not exist".format(val_dir))
        exit(0)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    time_start = time.time()

    class_file = {}
    val_list = []
    test_list = []
    class_dir_list = os.listdir(val_dir)
    for class_id in class_dir_list:
        class_dir = os.path.join(val_dir, class_id)
        file_list = os.listdir(class_dir)

        for file_id in file_list:
            if len(file_id) < 4 or file_id[-3:] not in ['jpg', 'png']:
                file_list.remove(file_id)
                continue
        val_list.extend([os.path.join(class_dir, file_id) for file_id in file_list])

    random.seed(random.randint(0, 100))
    random.shuffle(val_list)

    test_list.extend(val_list[:2000])

    # test
    for test_id in test_list:
        test_name = os.path.basename(test_id)
        class_id = os.path.basename(os.path.dirname(test_id))
        test_class_dir = os.path.join(test_dir, class_id)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        result_path = os.path.join(test_class_dir, test_name)
        shutil.copy(test_id, result_path)

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))