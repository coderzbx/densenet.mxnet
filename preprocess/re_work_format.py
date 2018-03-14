# -*-coding:utf-8-*-

import os
import time
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--augments', type=str, required=True)
    parser.add_argument('--check', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    start_t = time.time()

    # read all original images
    origin_map = {}
    src_dir = args.dir
    dir_list = os.listdir(src_dir)
    for dir_ in dir_list:
        if not dir_.isdigit():
            continue
        dir_path = os.path.join(src_dir, dir_)
        file_list = os.listdir(dir_path)

        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue
            origin_map[file_id] = int(dir_)

    # read augments images
    if not os.path.exists(args.augments):
        print("dir:{} not exist".format(args.augments))
        exit(0)
    augments_dir = args.augments
    dir_list = os.listdir(augments_dir)
    augments_map = {}
    for dir_ in dir_list:
        if not dir_.isdigit():
            continue
        dir_path = os.path.join(src_dir, dir_)
        file_list = os.listdir(dir_path)

        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue
            augments_map[file_id] = int(dir_)

    check_dir = args.check
    if not os.path.exists(check_dir):
        print("dir:{} not exist".format(check_dir))
        exit(0)

    check_map = {}
    dirs = ['same.low', 'diff.high', 'diff.low']
    for dir_ in dirs:
        dir_path = os.path.join(check_dir, dir_)
        csv_path = os.path.join(dir_path, "ImageType.csv")

        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r") as f:
            # skip first line
            line_str = f.readline()
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                image_name, class_id = line_str.split(",")
                check_map[image_name] = int(class_id)
                line_str = f.readline()

    # move wrong labels
    for image_name, class_id in check_map.items():
        if image_name in origin_map:
            origin_id = origin_map[image_name]
            if origin_id != class_id:
                src_path = os.path.join(src_dir, str(origin_id), image_name)
                dest_path = os.path.join(src_dir, str(class_id), image_name)

                shutil.move(src=src_path, dst=dest_path)
        elif image_name in augments_map:
            origin_id = augments_map[image_name]
            if origin_id != class_id:
                src_path = os.path.join(augments_dir, str(origin_id), image_name)
                dest_path = os.path.join(augments_dir, str(class_id), image_name)

                shutil.move(src=src_path, dst=dest_path)
        else:
            dest_path = os.path.join(src_dir, str(class_id), image_name)
            dirs = ['same.low', 'diff.high', 'diff.low']
            for dir_ in dirs:
                src_path = os.path.join(check_dir, dir_, image_name)
                if os.path.exists(src_path):
                    shutil.copy(src=src_path, dst=dest_path)
                    break

            print(image_name)

    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))

