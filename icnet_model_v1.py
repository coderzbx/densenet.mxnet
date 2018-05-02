from network_v2 import Network
import mxnet as mx


class ICNet(Network):
    def setup(self, is_training, num_classes, evalutaion):
        cudnn_off = False

        (self.feed('data')
             .interp(s_factor=0.5, name='data_sub2')
             .convolution(kernel=3, filters=32, stride=2, pad=1, no_bias=False, padding='SAME',
                          relu=True, name='conv1_1_3x3_s2', cudnn_off=cudnn_off)
             .convolution(kernel=3, filters=32, stride=1, pad=1, no_bias=False, padding='SAME',
                          relu=True, name='conv1_2_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=3, filters=64, stride=1, pad=1, no_bias=False, padding='SAME',
                          relu=True, name='conv1_3_3x3', cudnn_off=cudnn_off)
             .zero_padding(paddings=1, name='padding0')
             .pool(kernel=3, stride=2, pad=1, pool_type='max', name='pool1_3x3_s2')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv2_1_1x1_proj', cudnn_off=cudnn_off))

        (self.feed('pool1_3x3_s2')
             .convolution(kernel=1, filters=32, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv2_1_1x1_reduce', cudnn_off=cudnn_off)
             .convolution(kernel=3, filters=32, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv2_1_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv2_1_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv2_1_1x1_proj',
                   'conv2_1_1x1_increase')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .convolution(kernel=1, filters=32, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv2_2_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=1, name='padding2')
             .convolution(kernel=3, filters=32, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv2_2_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv2_2_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .convolution(kernel=1, filters=32, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv2_3_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=1, name='padding3')
             .convolution(kernel=3, filters=32, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv2_3_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv2_3_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .convolution(kernel=1, filters=256, stride=2, pad=0, no_bias=False,
                          relu=False, name='conv3_1_1x1_proj', cudnn_off=cudnn_off))

        (self.feed('conv2_3/relu')
             .convolution(kernel=1, filters=64, stride=2, pad=0, no_bias=False,
                          relu=True, name='conv3_1_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=1, name='padding4')
             .convolution(kernel=3, filters=64, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv3_1_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv3_1_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv3_1_1x1_proj',
                   'conv3_1_1x1_increase')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .interp(s_factor=0.5, name='conv3_1_sub4')
             .convolution(kernel=1, filters=64, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv3_2_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=1, name='padding5')
             .convolution(kernel=3, filters=64, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv3_2_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv3_2_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv3_1_sub4',
                   'conv3_2_1x1_increase')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .convolution(kernel=1, filters=64, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv3_3_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=1, name='padding6')
             .convolution(kernel=3, filters=64, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv3_3_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv3_3_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .convolution(kernel=1, filters=64, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv3_4_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=1, name='padding7')
             .convolution(kernel=3, filters=64, stride=1, pad=1, no_bias=False,
                          relu=True, name='conv3_4_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv3_4_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv4_1_1x1_proj', cudnn_off=cudnn_off))

        (self.feed('conv3_4/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv4_1_1x1_reduce', cudnn_off=cudnn_off)
             .zero_padding(paddings=2, name='padding8')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=True, name='conv4_1_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv4_1_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv4_1_1x1_proj',
                   'conv4_1_1x1_increase')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv4_2_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=2, name='padding9')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=True, name='conv4_2_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv4_2_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                   relu=True, name='conv4_3_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=2, name='padding10')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=True, name='conv4_3_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                   relu=False, name='conv4_3_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv4_4_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=2, name='padding11')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=True, name='conv4_4_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv4_4_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv4_5_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=2, name='padding12')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=True, name='conv4_5_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv4_5_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv4_6_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=2, name='padding13')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=True, name='conv4_6_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=512, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv4_6_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .convolution(kernel=1, filters=1024, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv5_1_1x1_proj', cudnn_off=cudnn_off))

        (self.feed('conv4_6/relu')
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv5_1_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=4, name='padding14')
             .convolution(kernel=3, filters=256, stride=4, pad=4, dilate=4, no_bias=False,
                          relu=True, name='conv5_1_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=1024, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv5_1_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv5_1_1x1_proj',
                   'conv5_1_1x1_increase')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv5_2_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=4, name='padding15')
             .convolution(kernel=3, filters=256, stride=4, dilate=4, no_bias=False,
                          relu=True, name='conv5_2_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=1024, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv5_2_1x1_increase', cudnn_off=cudnn_off))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv5_3_1x1_reduce', cudnn_off=cudnn_off)
             # .zero_padding(paddings=4, name='padding16')
             .convolution(kernel=3, filters=256, stride=4, pad=4, dilate=4, no_bias=False, 
                          relu=True, name='conv5_3_3x3', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=1024, stride=1, pad=0, no_bias=False, relu=False, name='conv5_3_1x1_increase'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        # shape = self.layers['conv5_3/relu'].get_shape().as_list()[1:3]
        arg_shape, output_shape, aux_shape = self.layers['conv5_3/relu'].infer_shape()
        shape = output_shape[0][2:]
        # shape = (mx.sym.Symbol.infer_shape(data=))[1]
        shape = (33, 65)
        h, w = shape

        # Change to same configuration as original prototxt
        if self.evaluation:
            (self.feed('conv5_3/relu')
                .pool(kernel=(33, 65), stride=(33, 65), name='conv5_3_pool1')
                .resize_bilinear(shape, name='conv5_3_pool1_interp'))

            (self.feed('conv5_3/relu')
                .pool(kernel=(17, 33), stride=(16, 32), name='conv5_3_pool2')
                .resize_bilinear(shape, name='conv5_3_pool2_interp'))

            (self.feed('conv5_3/relu')
                .pool(kernel=(13, 25), stride=(10, 20), name='conv5_3_pool3')
                .resize_bilinear(shape, name='conv5_3_pool3_interp'))

            (self.feed('conv5_3/relu')
                .pool(kernel=(8, 15), stride=(5, 10), name='conv5_3_pool6')
                .resize_bilinear(shape, name='conv5_3_pool6_interp'))
        else:       # In inference phase, we support different size of images as input.
            (self.feed('conv5_3/relu')
                .pool(kernel=(h, w), stride=(h, w), name='conv5_3_pool1')
                .resize_bilinear(shape, name='conv5_3_pool1_interp'))

            (self.feed('conv5_3/relu')
                .pool(kernel=(h/2, w/2), stride=(h/2, w/2), name='conv5_3_pool2')
                .resize_bilinear(shape, name='conv5_3_pool2_interp'))

            (self.feed('conv5_3/relu')
                .pool(kernel=(h/3, w/3), stride=(h/3, w/3), name='conv5_3_pool3')
                .resize_bilinear(shape, name='conv5_3_pool3_interp'))

            (self.feed('conv5_3/relu')
                .pool(kernel=(h/6, w/6), stride=(h/6, w/6), name='conv5_3_pool6')
                .resize_bilinear(shape, name='conv5_3_pool6_interp'))

        (self.feed('conv5_3/relu',
                   'conv5_3_pool6_interp',
                   'conv5_3_pool3_interp',
                   'conv5_3_pool2_interp',
                   'conv5_3_pool1_interp')
             .add(name='conv5_3_sum')
             .convolution(kernel=1, filters=256, stride=1, pad=0, no_bias=False,
                          relu=True, name='conv5_4_k1', cudnn_off=cudnn_off)
             .interp(z_factor=2.0, name='conv5_4_interp')
             # .zero_padding(paddings=2, name='padding17')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False,
                          relu=False, name='conv_sub4', cudnn_off=cudnn_off))

        (self.feed('conv3_1/relu')
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv3_1_sub2_proj', cudnn_off=cudnn_off))

        (self.feed('conv_sub4',
                   'conv3_1_sub2_proj')
             .add(name='sub24_sum')
             .relu(name='sub24_sum/relu')
             .interp(z_factor=2.0, name='sub24_sum_interp')
             # .zero_padding(paddings=2, name='padding18')
             .convolution(kernel=3, filters=128, stride=2, pad=2, dilate=2, no_bias=False, 
                          relu=False, name='conv_sub2', cudnn_off=cudnn_off))

        (self.feed('data')
             .convolution(kernel=3, filters=32, stride=2, pad=1, no_bias=False, padding='SAME',
                          relu=True, name='conv1_sub1', cudnn_off=cudnn_off)
             .convolution(kernel=3, filters=32, stride=2, pad=1, no_bias=False, padding='SAME',
                          relu=True, name='conv2_sub1', cudnn_off=cudnn_off)
             .convolution(kernel=3, filters=64, stride=2, pad=1, no_bias=False, padding='SAME',
                          relu=True, name='conv3_sub1', cudnn_off=cudnn_off)
             .convolution(kernel=1, filters=128, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv3_sub1_proj', cudnn_off=cudnn_off))

        (self.feed('conv_sub2',
                   'conv3_sub1_proj')
             .add(name='sub12_sum')
             .relu(name='sub12_sum/relu')
             .interp(z_factor=2.0, name='sub12_sum_interp')
             .convolution(kernel=1, filters=num_classes, stride=1, pad=0, no_bias=False,
                          relu=False, name='conv6_cls', cudnn_off=cudnn_off))
