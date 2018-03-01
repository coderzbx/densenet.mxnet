# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import random
from PIL import Image
# from tqdm import tqdm
import sklearn.cluster as cluster


def iou(x, centroids):
    dists = []
    for centroid in centroids:
        c_w, c_h = centroid
        w, h = x
        if c_w >= w and c_h >= h:
            dist = w * h / (c_w * c_h)
        elif c_w >= w and c_h <= h:
            dist = w * c_h / (w * h + (c_w - w) * c_h)
        elif c_w <= w and c_h >= h:
            dist = c_w * h / (w * h + c_w * (c_h - h))
        else:  # means both w,h are bigger than c_w and c_h respectively
            dist = (c_w * c_h) / (w * h)
        dists.append(dist)
    return np.array(dists)


def avg_iou(x, centroids):
    n, d = x.shape
    sums = 0.
    cluster_size = np.zeros((len(centroids),))
    cluster_items = []
    for i in range(len(centroids)):
      cluster_items.append([])
    for i in range(x.shape[0]):
        # note IOU() will return array which contains IoU for each centroid and X[i]
        # slightly ineffective, but I am too lazy
        dists = iou(x[i], centroids)
        cluster_size[np.argmax(dists)] += 1
        cluster_items[np.argmax(dists)].append(x[i])
        sums += max(dists)
    for i, items in enumerate(cluster_items):
      print(i, len(items))
      if len(items) < 10:
        print(str(items))
    return sums / n, cluster_size


def write_anchors_to_file(centroids, distance, cluster_size, anchor_file):
    # anchors = centroids * 416 / 32      # I do not know why it is 416/32
    anchors = centroids
    anchors = [str(i) for i in anchors.ravel()]
    print(
        "\n",
        "Cluster Result:\n",
        "Clusters:", len(centroids), "\n",
        "Clusters-zie:", cluster_size.astype(np.single), "\n",
        "Average IoU:", distance, "\n",
        "Anchors:\n", ", ".join(anchors)
    )

    with open(anchor_file, 'w') as f:
        f.write(", ".join(anchors))
        f.write('\n%f\n' % distance)


def k_means(x, n_clusters, eps):
    init_index = [random.randrange(x.shape[0]) for _ in range(n_clusters)]
    centroids = x[init_index]

    d = old_d = []
    iterations = 0
    diff = 1e10
    c, dim = centroids.shape

    while True:
        iterations += 1
        d = np.array([1 - iou(i, centroids) for i in x])
        if len(old_d) > 0:
            diff = np.sum(np.abs(d - old_d))

        print('diff = %f' % diff)

        if diff < eps or iterations > 1000:
            print("Number of iterations took = %d" % iterations)
            print("Centroids = ", centroids)
            return centroids

        # assign samples to centroids
        belonging_centroids = np.argmin(d, axis=1)

        # calculate the new centroids
        centroid_sums = np.zeros((c, dim), np.float)
        for i in range(belonging_centroids.shape[0]):
            centroid_sums[belonging_centroids[i]] += x[i]

        for j in range(c):
            centroids[j] = centroid_sums[j] / np.sum(belonging_centroids == j)

        old_d = d.copy()


def get_file_content(fnm):
    with open(fnm) as f:
        return [line.strip() for line in f]


def make_file(file_list, file_path):
    with open(file_path, "w") as f:
        with open(file_list, "r") as fin:
            while True:
                line = fin.readline()
                if not line:
                    break
                line = [i.strip() for i in line.strip().split('\t')]
                line_len = len(line)
                if line_len < 3:
                    print('lst should at least has three parts, but only has %s parts for %s' % (line_len, line))
                    continue
                file_path = line[2]
                image = Image.open(file_path)
                w, h = image.size
                f.write("{}\t{}\n".format(w, h))


def main(args):
    if not os.path.exists(args.file_list):
        print("file:{} is not exist".format(args.file_list))
        exit(0)

    # generate the anchor text
    if not os.path.exists("./anchor.txt"):
        make_file(args.file_list, "./anchor.txt")

    print("Reading Data ...")
    data = []
    for line in file("./anchor.txt"):
      line = line.strip().split("\t")
      # if float(line[-2]) > 1000:
      #   continue
      data.append((float(line[-2]), float(line[-1])))
    data = np.array(data).astype(np.single)
    print("data_shape: {}".format(data.shape))

    if args.engine.startswith("sklearn"):
        if args.engine == "sklearn":
            km = cluster.KMeans(n_clusters=args.num_clusters, tol=args.tol, verbose=True)
        elif args.engine == "sklearn-mini":
            km = cluster.MiniBatchKMeans(n_clusters=args.num_clusters, tol=args.tol, verbose=True)
        km.fit(data)
        result = km.cluster_centers_
        # distance = km.inertia_ / data.shape[0]
        distance, cluster_size = avg_iou(data, result)
    else:
        result = k_means(data, args.num_clusters, args.tol)
        distance, cluster_size = avg_iou(data, result)

    write_anchors_to_file(result, distance, cluster_size, args.output)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
#parser.add_argument('file_list', nargs='+', help='TrainList')
    parser.add_argument('--file_list', type=str, help='train list')
    parser.add_argument('--num_clusters', '-n', default=10, type=int, help='Number of Clusters')
    parser.add_argument('--output', '-o', default='./anchor_cluster.txt', type=str, help='Result Output File')
    parser.add_argument('--tol', '-t', default=0.00005, type=float, help='Tolerate')
    parser.add_argument('--engine', '-m', default='sklearn', type=str,
                        choices=['original', 'sklearn', 'sklearn-mini'], help='Method to use')

    args = parser.parse_args()

    main(args)

