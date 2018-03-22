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
    target_count = 500

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    class_images = {}

    dir_list = os.listdir(args.dir)
    for dir_id in dir_list:
        if not str(dir_id).isdigit():
            continue

        dir_path = os.path.join(args.dir, dir_id)
        file_list = os.listdir(dir_path)

        for file_name in file_list:
            if not file_name.endswith("jpg"):
                continue
            pos = file_name.find("r")
            if pos > 0:
                os.remove(os.path.join(args.dir, dir_id, file_name))
                continue

            file_path = os.path.join(args.dir, dir_id, file_name)
            class_id = dir_id
            if not class_id in class_images:
                class_images[class_id] = [file_path]
            else:
                class_images[class_id].append(file_path)
    for class_id, file_list in class_images.items():
        print("{}:{}".format(class_id, len(file_list)))

    start_t = time.time()
    for class_id, file_list in class_images.items():
        result_list = []
        if len(file_list) > 2000:
            # shuffle
            random.seed(random.random())
            random.shuffle(file_list)

            result_list.extend(file_list[:2000])
            for file_path in file_list[2000:]:
                os.remove(file_path)
        elif len(file_list) < 500:
            print("{}:len({}) need to augment -> 500".format(class_id, len(file_list)))
            result_list.extend(file_list[:])

            chosen_file = {}
            while len(result_list) < 500:
                random.seed(random.random())
                rotate_angle = random.uniform(rotate_range[0], rotate_range[1])
                print("rotate angle: {}".format(rotate_angle))

                random.seed(random.random())
                random.shuffle(file_list)

                file_path = file_list[0]
                image = Image.open(file_path)
                image_rotate = image.rotate(rotate_angle)

                class_dir = os.path.join(args.dir, class_id)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                result_name = os.path.basename(file_path)
                if not result_name in chosen_file:
                    chosen_file[result_name] = 1
                else:
                    chosen_file[result_name] += 1
                result_name = "{}_r_{}{}".format(result_name[:-4], chosen_file[result_name], result_name[-4:])
                result_path = os.path.join(class_dir, result_name)
                image_rotate.save(result_path)
                result_list.append(result_path)
    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))

