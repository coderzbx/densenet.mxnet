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
    parser.add_argument('--dir', help='path to folder saving packages.')
    parser.add_argument('--arrow_dir', help='path to arrow images.')
    parser.add_argument('--other_dir', help='path to other images.')

    args = parser.parse_args()

    root_dir = args.dir
    if not os.path.exists(root_dir):
        print("dir[{}] is not exist".format(root_dir))
        exit(0)

    arrow_dir = args.arrow_dir
    if not os.path.exists(arrow_dir):
        print("arrow_dir[{}] is not exist".format(arrow_dir))
        exit(0)

    other_dir = args.other_dir
    if not os.path.exists(other_dir):
        print("other_dir[{}] is not exist".format(other_dir))
        exit(0)

    time_start = time.time()
    step = 1000

    next_package = -1
    package_list = os.listdir(root_dir)
    for package_dir in package_list:
        if not package_dir.isdigit():
            continue

        if int(package_dir) > next_package:
            next_package = int(package_dir)

    root_dir = os.path.join(root_dir, "new_task")
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    # next_package += 1
    # print("arrow package start from {}\n".format(next_package))
    #
    # arrow_list = []
    # tmp_image_list = os.listdir(arrow_dir)
    # for image_name in tmp_image_list:
    #     name_list = image_name.split(".")
    #     if len(name_list) != 2:
    #         continue
    #     name_ext = name_list[1]
    #     if name_ext not in ["png", "jpg"]:
    #         continue
    #
    #     arrow_list.append(image_name)
    #
    # package_list = []
    #
    # for image_name in arrow_list:
    #     if len(package_list) < step:
    #         package_list.append(image_name)
    #     else:
    #         package_dir = os.path.join(root_dir, str(next_package))
    #         if not os.path.exists(package_dir):
    #             os.mkdir(package_dir)
    #         for id_ in package_list:
    #             src_path = os.path.join(arrow_dir, id_)
    #             dest_path = os.path.join(package_dir, id_)
    #             shutil.move(src_path, dest_path)
    #
    #         package_list = []
    #         next_package += 1
    #
    # if len(package_list) > 0:
    #     package_dir = os.path.join(root_dir, str(next_package))
    #     if not os.path.exists(package_dir):
    #         os.mkdir(package_dir)
    #     for id_ in package_list:
    #         src_path = os.path.join(arrow_dir, id_)
    #         dest_path = os.path.join(package_dir, id_)
    #         shutil.move(src_path, dest_path)
    #
    #     package_list = []
    #     next_package += 1
    #
    # print("arrow package end from {}".format(next_package-1))

    other_list = []
    next_package = 209
    print("other package start from {}\n".format(next_package))
    tmp_image_list = os.listdir(other_dir)
    for image_name in tmp_image_list:
        name_list = image_name.split(".")
        if len(name_list) != 2:
            continue
        name_ext = name_list[1]
        if name_ext not in ["png", "jpg"]:
            continue

        other_list.append(image_name)

    random.seed(100)
    random.shuffle(other_list)

    package_count = 0
    package_list = []
    for image_name in other_list:
        if len(package_list) < step:
            package_list.append(image_name)
        else:
            package_dir = os.path.join(root_dir, str(next_package))
            if not os.path.exists(package_dir):
                os.mkdir(package_dir)
            for id_ in package_list:
                src_path = os.path.join(other_dir, id_)
                dest_path = os.path.join(package_dir, id_)
                shutil.move(src_path, dest_path)

            package_list = []
            next_package += 1

            package_count += 1
            if package_count == 20:
                break

    if len(package_list) > 0:
        package_dir = os.path.join(root_dir, str(next_package))
        if not os.path.exists(package_dir):
            os.mkdir(package_dir)
        for id_ in package_list:
            src_path = os.path.join(other_dir, id_)
            dest_path = os.path.join(package_dir, id_)
            shutil.move(src_path, dest_path)

        package_list = []
        next_package += 1
    print("other package end from {}".format(next_package - 1))

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))