# -*-coding:utf-8-*-
import sys

sys.path.insert(0, "/opt/densenet.mxnet")

import os
import time
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=str, required=True)
    parser.add_argument('--done_dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.done_dir):
        print("dir[{}] is not exist".format(args.done_dir))
        exit(0)

    time_start = time.time()

    done_dir = args.done_dir
    done_list = os.listdir(done_dir)
    for done_id in done_list:
        name_list = done_id.split(".")
        name_ext = name_list[1]
        if name_ext not in ["jpg", "png"]:
            done_list.remove(done_id)

    package_list = os.listdir(args.package)
    for package_dir in package_list:
        if not package_dir.isdigit():
            continue
        label_file = os.path.join(args.package, package_dir, "ImageType.csv")
        if not os.path.exists(label_file):
            continue

        with open(label_file, "r") as f:
            line_str = f.readline()
            # skip first line
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                file_name, class_id = line_str.split(",")

                src_path = os.path.join(args.package, package_dir, file_name)
                class_dir = os.path.join(done_dir, class_id)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                dest_path = os.path.join(class_dir, file_name)
                shutil.copy(src_path, dest_path)

                line_str = f.readline()

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))