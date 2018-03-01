# -*-coding:utf-8-*-

import os
import time
import argparse
import shutil
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Static label data')
    parser.add_argument('--dir', help='path to folder saving packages.')
    args = parser.parse_args()

    root_dir = args.dir
    if not os.path.exists(root_dir):
        print("dir[{}] is not exist".format(root_dir))
        exit(0)

    time_start = time.time()
    step = 1000

    done_csv = os.path.join(root_dir, "ImageType.csv")
    if not os.path.exists(done_csv):
        print("file[{}] is not exist".format(done_csv))
        exit(0)

    dest_dir = os.path.join(root_dir, "../done_class")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(done_csv, "r") as f:
        line_str = f.readline()
        line_str = line_str.strip()

        while line_str:
            line_str = line_str.strip()
            file_name, class_id = line_str.split(",")

            class_dir = os.path.join(dest_dir, class_id)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            src_path = os.path.join(root_dir, file_name)
            dest_path = os.path.join(class_dir, file_name)
            shutil.copy(src_path, dest_path)

            line_str = f.readline()

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))