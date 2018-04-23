"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
from __future__ import print_function

import argparse
import os
import sys
import time
import mxnet as mx

import numpy as np

from icnet_model_v2 import ICNet_BN

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# If you want to apply to other datasets, change following four lines
DATA_DIR = '/PATH/TO/CITYSCAPES_DATASET'
DATA_LIST_PATH = './list/cityscapes_train_list.txt'
IGNORE_LABEL = 255  # The class number of background
INPUT_SIZE = '720, 720'  # Input size for training

BATCH_SIZE = 16
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 60001
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 50

# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0


def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    """Create the model and start the training."""
    data = mx.sym.Variable(name='data', shape=(1, 3, 1024, 2048))

    net = ICNet_BN({'data': data}, is_training=True, num_classes=13,
                   filter_scale=1)

    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_interp']
    all_inputs = sub124_out.list_arguments()

    layer_names = []
    for i in all_inputs:
        name_parts = str(i).split("_")
        if len(name_parts) > 1:
            name_parts = name_parts[:-1]
        # if str(i).endswith("weight") or str(i).endswith("bias") or str(i).endswith("gamma") or str(i).endswith("beta"):
        #     continue
        layer_name_ = "_".join(name_parts)
        if layer_name_ not in layer_names:
            layer_names.append(layer_name_)

    for layer_name_ in layer_names:
        print("Layer:{}".format(layer_name_))
        layer_ = net.layers[layer_name_]
        arg_shape, output_shape, aux_shape = layer_.infer_shape()
        print("Shape{}".format(output_shape))

    result_sym = net.get_output()
    mx.viz.plot_network(symbol=sub124_out).view()

if __name__ == '__main__':
    main()
