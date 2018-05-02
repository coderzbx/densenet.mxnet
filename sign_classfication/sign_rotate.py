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

    rotate_range = [-10, 10]
    target_count = 200

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    class_images = {}
    dir_list = os.listdir(args.dir)
    start_t = time.time()

    for dir_id in dir_list:
        dir_path = os.path.join(args.dir, dir_id)
        if not os.path.isdir(dir_path):
            continue

        file_list = os.listdir(dir_path)
        origin_list = []
        for file_id in file_list:
            if not file_id.endswith("jpg"):
                continue
            if file_id.find('r') >= 0:
                os.remove(os.path.join(dir_path, file_id))
                continue
            origin_list.append(file_id)
        cur_len = len(origin_list)
        print("{}:{}".format(dir_id, cur_len))
        rotate_map = {}
        while cur_len < 200:
            # rotate
            rotate_angle = random.uniform(rotate_range[0], rotate_range[1])
            print("angle:{}".format(rotate_angle))

            # random.seed(random.randint(10, 100))
            random.shuffle(origin_list)

            file_id = origin_list[0]
            file_path = os.path.join(dir_path, file_id)
            if not origin_list[0] in rotate_map:
                rotate_map[file_id] = 1
            else:
                rotate_map[file_id] += 1

            image = Image.open(file_path)
            image_rotate = image.rotate(rotate_angle)

            result_name = "{}_r_{}.jpg".format(file_id[:-4], rotate_map[file_id])
            result_path = os.path.join(dir_path, result_name)
            image_rotate.save(result_path)

            cur_len += 1

    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))

