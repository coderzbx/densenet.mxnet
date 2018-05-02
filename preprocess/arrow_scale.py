# -*-coding:utf-8-*-

import os
import time
import random
import argparse
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--result', type=str, required=True)
    args = parser.parse_args()

    scale_range = [0.8, 1.2]
    target_count = 1000

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)

    result_dir = args.result
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    class_images = {}

    all_type = os.path.join(args.dir, "ImageType.csv")
    if not os.path.exists(all_type):
        print("file:{} not exist".format(all_type))
    with open(all_type, "r") as f:
        line_str = f.readline()
        # skip first line
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()
            file_name, class_id = line_str.split(",")

            file_path = os.path.join(args.dir, file_name)
            if not class_id in class_images:
                class_images[class_id] = [file_path]
            else:
                class_images[class_id].append(file_path)

            line_str = f.readline()
    for class_id, file_list in class_images.items():
        print("{}:{}".format(class_id, len(file_list)))

    start_t = time.time()
    for class_id, file_list in class_images.items():
        result_list = []
        if len(file_list) > 5000:
            # shuffle
            random.seed(100)
            random.shuffle(file_list)

            result_list.extend(file_list[:5000])
        elif len(file_list) < 1000:
            print("{}:len({}) need to augment -> 1000".format(class_id, len(file_list)))
            result_list.extend(file_list[0:])

            chosen_file = {}
            while len(result_list) < 1000:

                scale_val = random.uniform(scale_range[0], scale_range[1])

                random.seed(random.randint(0, 100))
                random.shuffle(file_list)

                file_path = file_list[0]
                image = Image.open(file_path)
                w, h = image.size
                new_w = int(w * scale_val)
                new_h = int(h * scale_val)
                image_scale = image.resize((new_w, new_h))

                class_dir = os.path.join(result_dir, class_id)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)
                result_name = os.path.basename(file_path)
                if not result_name in chosen_file:
                    chosen_file[result_name] = 1
                else:
                    chosen_file[result_name] += 1
                result_name = "{}_s_{}{}".format(result_name[:-4], chosen_file[result_name], result_name[-4:])
                result_path = os.path.join(class_dir, result_name)
                image_scale.save(result_path)
                result_list.append(result_path)
    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))
