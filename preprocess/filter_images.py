# -*-coding:utf-8-*-

import os
import time
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    class_images = {}

    start_t = time.time()
    dir_list = os.listdir(args.dir)
    for dir_id in dir_list:
        if not dir_id.isdigit():
            continue

        dir_path = os.path.join(args.dir, dir_id)
        file_list = os.listdir(dir_path)

        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue

            file_path = os.path.join(dir_path, file_id)
            if file_id.find("r") > 0:
                os.remove(file_path)
                continue

            image = Image.open(file_path)

            w, h = image.size

            if w < 100 or h < 100:
                os.remove(file_path)

    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))

