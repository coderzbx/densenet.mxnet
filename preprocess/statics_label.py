# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import argparse
import time
from PIL import Image

from class_map import arrow_labels_v1


def static_label(args):
    time_start = time.time()

    class_id_map = {label.id: label.name for label in arrow_labels_v1}

    label_info = {}
    all_info = {}
    all_count = 0
    package_count = 0
    dir_list = os.listdir(args.dir)

    other_list = []
    label_list = []
    all_list = []

    for dir_name in dir_list:
        dir_path = os.path.join(args.dir, dir_name)
        if str(dir_name).isdigit() and os.path.isdir(dir_path):
            label_file = os.path.join(dir_path, "ImageType.csv")
            if not os.path.exists(label_file):
                continue
        else:
            continue

        package_count += 1

        total_count = 0
        with open(label_file, "r") as f:
            line_str = f.readline()
            # skip first line
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                image_name, class_id = line_str.split(",")

                class_id = int(class_id)

                if class_id not in label_info:
                    label_info[class_id] = 1
                else:
                    label_info[class_id] += 1

                if class_id not in all_info:
                    all_info[class_id] = 1
                else:
                    all_info[class_id] += 1

                image_path = os.path.join(args.dir, dir_name, image_name)
                if class_id == 0:
                    other_list.append(image_path)
                else:
                    label_list.append(image_path)
                all_list.append(image_path)

                total_count += 1
                all_count += 1
                line_str = f.readline()
        print("\npackage:{}-total:{}".format(dir_name, total_count))

        label_sort = sorted(label_info.items(), key=lambda d: d[0], reverse=False)
        for id, count in label_sort:
            print("{}-{}:{}".format(id, class_id_map[id].encode("utf-8"), count))

    print("\nall package:{}-total:{}".format(package_count, all_count))
    all_sort = sorted(all_info.items(), key=lambda d: d[0], reverse=False)
    for id, count in all_sort:
        print("{}-{}:{}".format(id, class_id_map[id].encode("utf-8"), count))

    with open("./anchor.txt", "w") as f:
        for image_path in all_list:
            image = Image.open(image_path)
            width = image.width
            height = image.height

            image_name = os.path.basename(image_path)
            out_str = "{}\t{}\t{}\n".format(image_name, width, height)
            f.write(out_str)

    shape_w_map = {}
    shape_h_map = {}
    shape_wh_map = {}

    total_other = len(other_list)
    for image_path in other_list:
        image = Image.open(image_path)
        width = image.width
        height = image.height

        if width <= 20:
            key = "width<=20"
        elif width <= 50:
            key = "20<=width<=50"
        elif width <= 100:
            key = "50<=width<=100"
        elif width <= 200:
            key = "100<=width<=200"
        elif width <= 300:
            key = "200<=width<=300"
        else:
            key = "width>=300"

        if key not in shape_w_map:
            shape_w_map[key] = 1
        else:
            shape_w_map[key] += 1

        if height <= 20:
            key = "height<=20"
        elif height <= 50:
            key = "20<=height<=50"
        elif height <= 100:
            key = "50<=height<=100"
        elif height <= 200:
            key = "100<=height<=200"
        elif height <= 300:
            key = "200<=height<=300"
        else:
            key = "height>=300"

        if key not in shape_h_map:
            shape_h_map[key] = 1
        else:
            shape_h_map[key] += 1

        if width <= 100 and height <= 100:
            key = "width,height<=100"
            if not key in shape_wh_map:
                shape_wh_map[key] = 1
            else:
                shape_wh_map[key] += 1
        elif width <= 200 and height <= 200:
            key = "width,height<=200"
            if not key in shape_wh_map:
                shape_wh_map[key] = 1
            else:
                shape_wh_map[key] += 1

    key = "width<=20"
    print("\n{} : {}/{}".format(key, shape_w_map[key], total_other))
    key = "20<=width<=50"
    print("{} : {}/{}".format(key, shape_w_map[key], total_other))
    key = "50<=width<=100"
    print("{} : {}/{}".format(key, shape_w_map[key], total_other))
    key = "100<=width<=200"
    print("{} : {}/{}".format(key, shape_w_map[key], total_other))
    key = "200<=width<=300"
    print("{} : {}/{}".format(key, shape_w_map[key], total_other))
    key = "width>=300"
    print("{} : {}/{}\n".format(key, shape_w_map[key], total_other))

    key = "height<=20"
    print("{} : {}/{}".format(key, shape_h_map[key], total_other))
    key = "20<=height<=50"
    print("{} : {}/{}".format(key, shape_h_map[key], total_other))
    key = "50<=height<=100"
    print("{} : {}/{}".format(key, shape_h_map[key], total_other))
    key = "100<=height<=200"
    print("{} : {}/{}".format(key, shape_h_map[key], total_other))
    key = "200<=height<=300"
    print("{} : {}/{}".format(key, shape_h_map[key], total_other))
    key = "height>=300"
    print("{} : {}/{}\n".format(key, shape_h_map[key], total_other))

    label_w_map = {}
    label_h_map = {}
    label_wh_map = {}

    total_label = len(label_list)
    for image_path in label_list:
        image = Image.open(image_path)
        width = image.width
        height = image.height

        if width <= 20:
            key = "width<=20"
        elif width <= 50:
            key = "20<=width<=50"
        elif width <= 100:
            key = "50<=width<=100"
        elif width <= 200:
            key = "100<=width<=200"
        elif width <= 300:
            key = "200<=width<=300"
        else:
            key = "width>=300"

        if key not in label_w_map:
            label_w_map[key] = 1
        else:
            label_w_map[key] += 1

        if height <= 20:
            key = "height<=20"
        elif height <= 50:
            key = "20<=height<=50"
        elif height <= 100:
            key = "50<=height<=100"
        elif height <= 200:
            key = "100<=height<=200"
        elif height <= 300:
            key = "200<=height<=300"
        else:
            key = "height>=300"

        if key not in label_h_map:
            label_h_map[key] = 1
        else:
            label_h_map[key] += 1

        if width <= 100 and height <= 100:
            key = "width,height<=100"
            if not key in label_wh_map:
                label_wh_map[key] = 1
            else:
                label_wh_map[key] += 1
        elif width <= 200 and height <= 200:
            key = "width,height<=200"
            if not key in label_wh_map:
                label_wh_map[key] = 1
            else:
                label_wh_map[key] += 1

    key = "width<=20"
    print("\n{} : {}/{}".format(key, label_w_map[key], total_other))
    key = "20<=width<=50"
    print("{} : {}/{}".format(key, label_w_map[key], total_other))
    key = "50<=width<=100"
    print("{} : {}/{}".format(key, label_w_map[key], total_other))
    key = "100<=width<=200"
    print("{} : {}/{}".format(key, label_w_map[key], total_other))
    key = "200<=width<=300"
    print("{} : {}/{}".format(key, label_w_map[key], total_other))
    key = "width>=300"
    print("{} : {}/{}\n".format(key, label_w_map[key], total_other))

    key = "height<=20"
    print("{} : {}/{}".format(key, label_h_map[key], total_other))
    key = "20<=height<=50"
    print("{} : {}/{}".format(key, label_h_map[key], total_other))
    key = "50<=height<=100"
    print("{} : {}/{}".format(key, label_h_map[key], total_other))
    key = "100<=height<=200"
    print("{} : {}/{}".format(key, label_h_map[key], total_other))
    key = "200<=height<=300"
    print("{} : {}/{}".format(key, label_h_map[key], total_other))
    key = "height>=300"
    print("{} : {}/{}\n".format(key, label_h_map[key], total_other))

    for key, value in shape_wh_map.items():
        print("{} : {}/{}\n".format(key, value, total_other))

    for key, value in label_wh_map.items():
        print("{} : {}/{}\n".format(key, value, total_label))

    time_end = time.time()
    print("finish in {} s".format(time_end-time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Static label data')
    parser.add_argument('--dir', help='path to folder containing images.')

    args = parser.parse_args()
    static_label(args)
