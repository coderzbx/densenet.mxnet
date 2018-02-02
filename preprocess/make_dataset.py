# -*-coding:utf-8-*-

import os
import time
import argparse
import shutil
import random


if __name__ == '__main__':
    # /data/deeplearning/dataset/label_arrow/done
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print("dir[{}] is not exist".format(args.done_dir))
        exit(0)

    time_start = time.time()

    done_dir = args.dir
    train_dir = os.path.join(done_dir, "train")
    val_dir = os.path.join(done_dir, "val")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    done_csv = os.path.join(done_dir, "ImageType.csv")
    if not os.path.exists(done_csv):
        list_path = "/data/deeplearning/dataset/label_arrow/training/20180202-cls15.lst"
        with open(done_csv, "w") as f:
            with open(list_path) as fin:
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    line = [i.strip() for i in line.strip().split('\t')]
                    label_info = "{},{}\n".format(os.path.basename(line[2]), int(line[1]))
                    f.write(label_info)
    csv_list = []
    with open(done_csv, "r") as f:
        line_str = f.readline()
        # skip first line
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()

            file_name, class_id = line_str.split(",")
            src_path = os.path.join(done_dir, file_name)
            if not os.path.exists(src_path):
                line_str = f.readline()
                continue

            csv_list.append(line_str)
            line_str = f.readline()

    # shuffle
    random.seed(100)
    random.shuffle(csv_list)

    val_count = int(len(csv_list) / 8)
    train_count = len(csv_list) - val_count

    i = 0
    train_list = []
    val_list = []
    for label_info in csv_list:
        if i < val_count:
            val_list.append(label_info)
        else:
            train_list.append(label_info)
        i += 1

    train_csv = os.path.join(train_dir, "ImageType.csv")
    with open(train_csv, "a+") as fw:
        for label_info in train_list:
            file_name, class_id = label_info.split(",")
            src_path = os.path.join(done_dir, file_name)
            dest_path = os.path.join(train_dir, file_name)
            shutil.copy(src_path, dest_path)
            fw.write(label_info + "\n")

    val_csv = os.path.join(val_dir, "ImageType.csv")
    with open(val_csv, "a+") as fw:
        for label_info in val_list:
            file_name, class_id = label_info.split(",")
            src_path = os.path.join(done_dir, file_name)
            dest_path = os.path.join(val_dir, file_name)
            shutil.copy(src_path, dest_path)
            fw.write(label_info + "\n")

    time_end = time.time()
    print("finish in {} s".format(time_end - time_start))