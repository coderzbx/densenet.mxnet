# -*-coding:utf-8-*-

import os
import shutil

if __name__ == "__main__":
    last_package = 198

    image_dir = "/data/pure-ftpd/data/rec_user"
    dir_list = os.listdir(image_dir)

    for dir_id in dir_list:
        if not dir_id.isdigit():
            continue

        if int(dir_id) < last_package:
            old_dir = os.path.join(image_dir, dir_id)
            new_dir = os.path.join(image_dir, "old_task", dir_id)
            shutil.move(old_dir, new_dir)