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


def static_label(args):
    time_start = time.time()

    all_info = {}
    all_count = 0
    dir_list = os.listdir(args.dir)

    all_list = []

    label_count_map = {}

    for dir_name in dir_list:
        dir_path = os.path.join(args.dir, dir_name)
        if os.path.isdir(dir_path):
            label_file = os.path.join(dir_path, "ImageType.csv")
            if not os.path.exists(label_file):
                continue
        else:
            continue

        total_count = 0
        with open(label_file, "r") as f:
            line_str = f.readline()
            # skip first line
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                image_name, class_id = line_str.split(",")

                if class_id not in label_count_map:
                    label_count_map[class_id] = 1
                else:
                    label_count_map[class_id] += 1

                if class_id not in all_info:
                    all_info[class_id] = 1
                else:
                    all_info[class_id] += 1

                image_path = os.path.join(args.dir, dir_name, image_name)
                all_list.append(image_path)

                total_count += 1
                all_count += 1
                line_str = f.readline()
        print("\npackage:{}-total:{}".format(dir_name, total_count))

        label_sort = sorted(label_count_map.items(), key=lambda d: d[1], reverse=False)
        for id, count in label_sort:
            print("{}:{}".format(id, count))

    print("\nall package-total:{}".format(all_count))
    all_sort = sorted(all_info.items(), key=lambda d: d[0], reverse=False)
    for id, count in all_sort:
        print("{}:{}".format(id, count))

    time_end = time.time()
    print("finish in {} s".format(time_end-time_start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Static label data')
    parser.add_argument('--dir', help='path to folder containing images.')

    args = parser.parse_args()
    static_label(args)
