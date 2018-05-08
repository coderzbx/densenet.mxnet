# -*-coding:utf-8-*-

import os
import time
import random
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    rotate_range = [-2, 2]
    target_count = 3000

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    class_images = {}

    dir_list = os.listdir(args.dir)
    for dir_id in dir_list:
        start_t = time.time()
        if not str(dir_id).isdigit():
            continue

        dir_path = os.path.join(args.dir, dir_id)
        file_list = os.listdir(dir_path)

        jpg_list = []
        cur_list = []
        chosen_file = {}
        for file_name in file_list:
            if not file_name.endswith("jpg"):
                continue
            pos = file_name.find("r")
            if pos > 0:
                os.remove(os.path.join(args.dir, dir_id, file_name))
                continue

            file_path = os.path.join(args.dir, dir_id, file_name)
            jpg_list.append(file_path)
            cur_list.append(file_path)
            chosen_file[os.path.basename(file_path)] = 0

        if len(jpg_list) < target_count:
            while len(cur_list) < target_count:
                random.shuffle(jpg_list)
                random.seed(random.random())
                rotate_angle = random.uniform(rotate_range[0], rotate_range[1])
                # print("rotate angle: {}".format(rotate_angle))

                file_path = jpg_list[0]
                image = Image.open(file_path)
                image_rotate = image.rotate(rotate_angle)

                result_name = os.path.basename(file_path)
                chosen_file[result_name] += 1

                result_name = "{}_r_{}.jpg".format(result_name[:-4], chosen_file[result_name])
                result_path = os.path.join(dir_path, result_name)
                image_rotate.save(result_path)
                cur_list.append(result_path)


        end_t = time.time()
        print("finish in {} s".format(end_t - start_t))
