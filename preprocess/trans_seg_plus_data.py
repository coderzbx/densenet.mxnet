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
import sys
sys.path.insert(0, "/opt/densenet.mxnet")

import mxnet as mx
import random
import argparse
import cv2
import numpy as np
import time
import traceback

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


def list_image(root, recursive, exts):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(root, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, root), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, root), v)
    else:
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, root), 0)
                i += 1


def write_list(path_out, image_list, rec_image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '{}\t'.format(j)
            label = rec_image_list[item[1]]
            line += '{}\t{}\n'.format(item[1], label)
            fout.write(line)


def make_list_dir(args):
    image_label_map = {}
    image_list = []

    dir_path = args.root
    image_index = 0

    package_dir_list = os.listdir(dir_path)
    for package_id in package_dir_list:
        if not package_id.isdigit():
            continue

        file_list = os.listdir(os.path.join(dir_path, package_id))
        for file_id in file_list:
            if file_id.startswith("label") or file_id.endswith("png"):
                continue

            image_path = os.path.join(dir_path, package_id, file_id)
            if file_id.endswith("jpg"):
                label_path = image_path[:-3] + "png"
                if os.path.exists(label_path):
                    image_list.append((image_index, image_path, "0"))
                    image_index += 1
                    image_label_map[image_path] = label_path
                else:
                    print(image_path)

    # image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(random.random())
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) // args.chunks
    for i in range(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        if args.train_ratio == 1.0:
            write_list(args.prefix + str_chunk + '.lst', chunk, image_label_map)
        else:
            if args.test_ratio:
                write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test], image_label_map)
            if args.train_ratio + args.test_ratio < 1.0:
                write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:], image_label_map)
            write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep], image_label_map)


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 4:
                print('lst should at least has three parts, but only has %s parts for %s' %(line_len, line))
                continue
            try:
                item = [int(line[0])] + [float(line[1])] + [i for i in line[2:]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' %(line, e))
                continue
            yield item


def image_encode(args, i, item, q_out):
    fullpath = item[2]

    if not os.path.exists(fullpath):
        print(fullpath)

    # if len(item) > 3 and args.pack_label:
    #     header = mx.seg_recordio.ISegRHeader(0, 0, 0, 0, item[0], 0)
    # else:
    #     header = mx.seg_recordio.ISegRHeader(0, 0, 0, 0, item[0], 0)

    if args.pass_through:
        try:
            img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
            ret, buf = cv2.imencode(".jpg", img)
            assert ret, 'failed to encode image'
            image_data = buf.tostring()
            image_len = len(image_data)

            label_path = item[-1]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            ret, buf = cv2.imencode(".png", label)
            assert ret, 'failed to encode label'
            label_data = buf.tostring()
            label_len = len(label_data)

            header = mx.seg_recordio.ISegRHeader(0, 0, image_len, label_len, item[0], 0)

            s = mx.seg_recordio.pack(header, image_data, label_data)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
        label_path = item[-1]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        ret, buf = cv2.imencode(".jpg", img)
        assert ret, 'failed to encode image'
        image_data = buf.tostring()
        image_len = len(image_data)

        ret, buf = cv2.imencode(".png", label)
        assert ret, 'failed to encode label'
        label_data = buf.tostring()
        label_len = len(label_data)
        # with open(fullpath, "r") as f:
        #     s = f.read()
        #     image_len = len(s)
        # with open(label_path, "r") as f1:
        #     s1 = f1.read()
        #     label_len = len(s1)

        header = mx.seg_recordio.ISegRHeader(0, 0, image_len, label_len, item[0], 0)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        q_out.put((i, None, item))
        return
    if args.center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) // 2
            img = img[margin:margin + img.shape[1], :]
            label = label[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2
            img = img[:, margin:margin + img.shape[0]]
            label = label[:, margin:margin + label.shape[0]]
    if args.center_pad:
        newsize = max(img.shape[:2])
        new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
        new_label = np.ones((newsize, newsize) + label.shape[2:], np.uint8) * 127
        margin0 = (newsize - img.shape[0]) // 2
        margin1 = (newsize - img.shape[1]) // 2
        new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
        new_label[margin0:margin0 + label.shape[0], margin1:margin1 + label.shape[1]] = label
        img = new_img
        label = new_label
    if args.resize:
        if img.shape[0] > img.shape[1]:
            newsize = (args.resize, img.shape[0] * args.resize // img.shape[1])
        else:
            newsize = (img.shape[1] * args.resize // img.shape[0], args.resize)
        img = cv2.resize(img, newsize)
        label = cv2.resize(label, newsize)

    try:
        s = mx.seg_recordio.pack_img(header, img, label, quality=args.quality, img_fmt='.jpg', label_fmt='.png')
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return


def read_worker(args, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(args, i, item, q_out)


def write_worker(q_out, fname, working_dir):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.seg_recordio.MXIndexedSegRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 100 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1


def parse_args():
    # python /opt/densenet.mxnet/preprocess/trans_arrow_data.py
    # /data/deeplearning/dataset/arrow/data_0315
    # /data/deeplearning/dataset/arrow/label_0314
    # --train-ratio=0.78
    # --test-ratio=0.07
    # --center-pad
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', action='store_true',
                        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', nargs='+', default=['.jpeg', '.jpg', '.png'],
                        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0,
                        help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0,
                        help='Ratio of images to use for testing.')
    cgroup.add_argument('--recursive', action='store_true',
                        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument('--no-shuffle', dest='shuffle', action='store_false',
                        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--pass-through', action='store_true',
                        help='whether to skip transformation and save image as is')
    rgroup.add_argument('--resize', type=int, default=0,
                        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center-crop', action='store_true',
                        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--center-pad', action='store_true',
                        help='specify whether to pad the whole image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=95,
                        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num-thread', type=int, default=1,
                        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=-1, choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
                        help='specify the encoding of the images.')
    rgroup.add_argument('--pack-label', action='store_true',
        help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    # args.prefix = os.path.abspath(args.prefix)
    # args.root = os.path.abspath(args.root)
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.list:
        make_list_dir(args)
    else:
        if os.path.isdir(args.prefix):
            working_dir = args.prefix
        else:
            working_dir = os.path.dirname(args.prefix)
        files = [os.path.join(working_dir, fname) for fname in os.listdir(working_dir)
                    if os.path.isfile(os.path.join(working_dir, fname))]
        count = 0
        for fname in files:
            if fname.startswith(args.prefix) and fname.endswith('.lst'):
                print('Creating .rec file from', fname, 'in', working_dir)
                count += 1
                image_list = read_list(fname)
                # -- write_record -- #
                if args.num_thread > 1 and multiprocessing is not None:
                    q_in = [multiprocessing.Queue(1024) for i in range(args.num_thread)]
                    q_out = multiprocessing.Queue(1024)
                    read_process = [multiprocessing.Process(target=read_worker, args=(args, q_in[i], q_out)) \
                                    for i in range(args.num_thread)]
                    for p in read_process:
                        p.start()
                    write_process = multiprocessing.Process(target=write_worker, args=(q_out, fname, working_dir))
                    write_process.start()

                    for i, item in enumerate(image_list):
                        q_in[i % len(q_in)].put((i, item))
                    for q in q_in:
                        q.put(None)
                    for p in read_process:
                        p.join()

                    q_out.put(None)
                    write_process.join()
                else:
                    print('multiprocessing not available, fall back to single threaded encoding')
                    try:
                        import Queue as queue
                    except ImportError:
                        import queue
                    q_out = queue.Queue()
                    fname = os.path.basename(fname)
                    fname_rec = os.path.splitext(fname)[0] + '.rec'
                    fname_idx = os.path.splitext(fname)[0] + '.idx'
                    record = mx.seg_recordio.MXIndexedSegRecordIO(os.path.join(working_dir, fname_idx),
                                                           os.path.join(working_dir, fname_rec), 'w')
                    cnt = 0
                    pre_time = time.time()
                    for i, item in enumerate(image_list):
                        image_encode(args, i, item, q_out)
                        if q_out.empty():
                            continue
                        _, s, _ = q_out.get()
                        record.write_idx(item[0], s)
                        if cnt % 100 == 0:
                            cur_time = time.time()
                            print('time:', cur_time - pre_time, ' count:', cnt)
                            pre_time = cur_time
                        cnt += 1
        if not count:
            print('Did not find and list file with prefix %s'%args.prefix)
