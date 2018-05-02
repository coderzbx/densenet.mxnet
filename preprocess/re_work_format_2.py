# -*-coding:utf-8-*-

import os
import time
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--check', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print("dir:{} not exist".format(args.dir))
        exit(0)
    src_dir = args.dir

    if not os.path.exists(args.label):
        print("dir:{} not exist".format(args.label))
        exit(0)
    label_dir = args.label

    check_dir = args.check
    if not os.path.exists(check_dir):
        print("dir:{} not exist".format(check_dir))
        exit(0)

    start_t = time.time()

    # formate same.high
    dir_path = os.path.join(check_dir, "same.high")
    csv_path = os.path.join(check_dir, "info.csv")
    with open(csv_path, "r") as f:
        # skip first line
        line_str = f.readline()
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()
            image_name, class_id, predict, score = line_str.split(",")
            score = float(score)
            if class_id == predict and score >= 0.95:
                dest_dir = os.path.join(src_dir, class_id)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                dest_path = os.path.join(dest_dir, image_name)
                src_path = os.path.join(dir_path, class_id, image_name)

                if not os.path.exists(src_path):
                    print(line_str)
                    src_path = os.path.join(label_dir, class_id, image_name)
                    if os.path.exists(src_path):
                        print(line_str)
                        shutil.copy(src_path, dest_path)
                else:
                    shutil.copy(src_path, dest_path)
            line_str = f.readline()

    dirs = ['same.low', 'diff.high', 'diff.low']
    for dir_ in dirs:
        dir_path = os.path.join(check_dir, dir_)
        csv_path = os.path.join(dir_path, "ImageType.csv")

        if not os.path.exists(csv_path):
            continue

        with open(csv_path, "r") as f:
            # skip first line
            line_str = f.readline()
            line_str = f.readline()
            while line_str:
                line_str = line_str.strip()
                image_name, class_id = line_str.split(",")

                dest_dir = os.path.join(src_dir, class_id)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                dest_path = os.path.join(dest_dir, image_name)
                src_path = os.path.join(dir_path, image_name)

                if not os.path.exists(src_path):
                    print(line_str)
                else:
                    shutil.copy(src_path, dest_path)

                line_str = f.readline()

    end_t = time.time()
    print("finish in {} s".format(end_t - start_t))

