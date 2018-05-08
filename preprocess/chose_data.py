# -*-coding:utf-8-*-

import os
import time
import random
import argparse
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()
    target_count = 6000

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    class_images = {}

    dir_list = os.listdir(args.dir)
    for dir_id in dir_list:
        start_t = time.time()
        if not str(dir_id).isdigit():
            continue

        dir_path = os.path.join(args.dir, dir_id)
        file_list = os.listdir(dir_path)

        jpg_list = []
        chosen_list = []
        for file_name in file_list:
            if not file_name.endswith("jpg"):
                continue
            pos = file_name.find("r")
            if pos > 0:
                os.remove(os.path.join(args.dir, dir_id, file_name))
                continue

            file_path = os.path.join(args.dir, dir_id, file_name)
            jpg_list.append(file_path)

        if len(jpg_list) < target_count:
            chosen_list = jpg_list[:]
        else:
            random.shuffle(jpg_list)
            chosen_list = jpg_list[:target_count]

        for file_path in chosen_list:
            if not os.path.exists(os.path.join(args.dest_dir, dir_id)):
                os.makedirs(os.path.join(args.dest_dir, dir_id))

            dest_path = os.path.join(args.dest_dir, dir_id, os.path.basename(file_path))
            shutil.copy(file_path, dest_path)

        end_t = time.time()
        print("finish in {} s".format(end_t - start_t))
