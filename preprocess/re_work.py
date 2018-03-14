# -*-coding:utf-8-*-

import os
import time
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    result_dir = args.task
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    class_images = {}
    dirs = ['same.high', 'same.low', 'diff.high', 'diff.low']

    check_info = os.path.join(args.dir, "info.csv")
    if not os.path.exists(check_info):
        print("file:{} not exist".format(check_info))

    check_list = {}

    start_t = time.time()

    with open(check_info, "r") as f:
        line_str = f.readline()
        # skip first line
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()

            # label_id: id labeled in dataset
            # class_id: id predict
            file_name, label_id, class_id, class_acc = line_str.split(",")
            class_acc = float(class_acc)

            if file_name == '327_20180115133328530498_00_004_5.jpg':
                print("got")

            init_id = class_id
            if label_id == class_id:
                if class_acc < 0.8:
                    init_id = label_id
                    _dir = 'same.low'
                elif class_acc < 0.95:
                    _dir = 'same.high'
                else:
                    line_str = f.readline()
                    continue
            else:
                if class_acc < 0.8:
                    init_id = label_id
                    _dir = 'diff.low'
                else:
                    _dir = 'diff.high'

            dest_dir = os.path.join(result_dir, _dir)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            if _dir not in check_list:
                check_list[_dir] = []

            task_str = "{},{}\n".format(file_name, init_id)
            check_list[_dir].append(task_str)

            dest_path = os.path.join(dest_dir, file_name)
            file_path = os.path.join(args.dir, _dir, class_id, file_name)

            shutil.copy(file_path, dest_path)

            line_str = f.readline()

    for _dir, _dir_list in check_list.items():
        csv_path = os.path.join(result_dir, _dir, "ImageType.csv")
        with open(csv_path, "w") as f:
            for _str in _dir_list:
                f.write(_str)

    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))

