# -*-coding:utf-8-*-

import os
import time
import argparse
import shutil
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.src_dir):
        print("dir[{}] is not exist".format(args.src_dir))
        exit(0)

    time_start = time.time()

    src_dir = args.src_dir
    dest_dir = args.dest_dir
    src_dir_list = os.listdir(src_dir)
    for dir_id in src_dir_list:
        if not str(dir_id).isdigit():
            continue

        dir_path = os.path.join(src_dir, dir_id)
        file_list = os.listdir(dir_path)

        for file_id in file_list:
            if not file_id.endswith("jpg"):
                file_list.remove(file_id)

        if len(file_list) > 10000:
            chose_count = int(len(file_list) / 2)
            if chose_count < 10000:
                chose_count = 10000

            random.seed(random.randint(0, 100))
            random.shuffle(file_list)

            chose_list = file_list[0: chose_count]
        else:
            chose_list = file_list

        for chose_id in chose_list:
            src_path = os.path.join(src_dir, dir_id, chose_id)
            dest_path = os.path.join(dest_dir, dir_id, chose_id)
            if not os.path.exists(os.path.dirname(dest_path)):
                os.makedirs(os.path.dirname(dest_path))
            shutil.copy(src_path, dest_path)

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))