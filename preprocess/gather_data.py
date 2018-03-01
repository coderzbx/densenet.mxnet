# -*-coding:utf-8-*-
import os
import shutil

src_dir = "/data/deeplearning/dataset/label_arrow/data/20180205/mask"
dest_dir = "/data/deeplearning/dataset/label_arrow/data/20180205/work/images"

if __name__ == "__main__":
    dir_list = os.listdir(src_dir)

    for track_dir in dir_list:
        if not track_dir.find("_"):
            continue

        dir_path = os.path.join(src_dir, track_dir)

        image_list = os.listdir(dir_path)
        for image_id in image_list:
            src_path = os.path.join(dir_path, image_id)
            dest_path = os.path.join(dest_dir, image_id)

            if src_path.endswith("jpg"):
                shutil.copy(src_path, dest_path)