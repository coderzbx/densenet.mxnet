# -*-coding:utf-8-*-
import os
import shutil

src_dir = "/data/deeplearning/dataset/arrow_data/work/other_pic"
dest_dir = "/data/deeplearning/dataset/arrow_data/work/all"

if __name__ == "__main__":
    image_list = os.listdir(src_dir)
    for image_id in image_list:
        src_path = os.path.join(src_dir, image_id)
        dest_path = os.path.join(dest_dir, image_id)
        shutil.move(src_path, dest_path)