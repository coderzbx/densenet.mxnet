# -*-coding:utf-8-*-

import os
import time
import argparse
import shutil
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Static label data')
    parser.add_argument('--package', help='path to folder saving packages.',
                        default="/data/deeplearning/dataset/arrow_data/work/new_task")
    parser.add_argument('--image', help='path to images.',
                        default="/data/deeplearning/dataset/label_arrow/data/20180205/work/images")

    args = parser.parse_args()

    root_dir = args.image
    if not os.path.exists(root_dir):
        print("dir[{}] is not exist".format(root_dir))
        exit(0)

    dest_package_dir = args.package
    if not os.path.exists(dest_package_dir):
        print("dir[{}] is not exist".format(dest_package_dir))
        exit(0)

    next_package = -1
    package_list = os.listdir(dest_package_dir)
    for package in package_list:
        if not package.isdigit():
            continue

        if int(package) > next_package:
            next_package = int(package)

    time_start = time.time()
    step = 1000

    next_package += 1
    print("arrow package start from {}\n".format(next_package))

    arrow_list = []
    other_list = []

    class_dir_list = os.listdir(root_dir)
    for class_id in class_dir_list:
        if not class_id.isdigit():
            continue
        class_dir = os.path.join(root_dir, class_id)

        tmp_image_list = os.listdir(class_dir)
        for image_name in tmp_image_list:
            name_list = image_name.split(".")
            if len(name_list) != 2:
                continue
            name_ext = name_list[1]
            if name_ext not in ["png", "jpg"]:
                continue

            src_path = os.path.join(class_dir, image_name)
            if int(class_id) in range(1, 12):
                # arrow class
                arrow_list.append(src_path)
            else:
                other_list.append(src_path)

    random.seed(100)
    random.shuffle(arrow_list)

    package_list = []
    for image_name in arrow_list:
        if len(package_list) < step:
            package_list.append(image_name)
        else:
            package_dir = os.path.join(dest_package_dir, str(next_package))
            if not os.path.exists(package_dir):
                os.mkdir(package_dir)
            for id_ in package_list:
                file_name = os.path.basename(id_)
                src_path = id_
                dest_path = os.path.join(package_dir, file_name)
                shutil.copy(src_path, dest_path)

            package_list = []
            next_package += 1

    if len(package_list) > 0:
        package_dir = os.path.join(dest_package_dir, str(next_package))
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)
        for id_ in package_list:
            file_name = os.path.basename(id_)
            src_path = id_
            dest_path = os.path.join(package_dir, file_name)
            shutil.copy(src_path, dest_path)

        package_list = []
        next_package += 1

    print("arrow package end from {}".format(next_package-1))

    print("other package start from {}\n".format(next_package))

    random.seed(100)
    random.shuffle(other_list)

    package_count = 0
    package_list = []
    for image_name in other_list:
        if len(package_list) < step:
            package_list.append(image_name)
        else:
            package_dir = os.path.join(dest_package_dir, str(next_package))
            if not os.path.exists(package_dir):
                os.mkdir(package_dir)
            for id_ in package_list:
                file_name = os.path.basename(id_)
                src_path = id_
                dest_path = os.path.join(package_dir, file_name)
                shutil.copy(src_path, dest_path)

            package_list = []
            next_package += 1

            package_count += 1
            if package_count == 20:
                break

    if len(package_list) > 0:
        package_dir = os.path.join(dest_package_dir, str(next_package))
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)
        for id_ in package_list:
            file_name = os.path.basename(id_)
            src_path = id_
            dest_path = os.path.join(package_dir, file_name)
            shutil.copy(src_path, dest_path)

        package_list = []
        next_package += 1
    print("other package end from {}".format(next_package - 1))

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))