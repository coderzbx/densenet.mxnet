import numpy as np
import mxnet as mx


DEFAULT_PADDING = 'VALID'
DEFAULT_DATAFORMAT = 'NHWC'
layer_name = []
BN_param_map = {'scale':    'gamma',
                'offset':   'beta',
                'variance': 'moving_variance',
                'mean':     'moving_mean'}

cfg = {}


def _attr_scope_lr(lr_type, lr_owner):
    assert lr_type in ('alex', 'alex10', 'torch')
    # weight (lr_mult, wd_mult); bias;
    # 1, 1; 2, 0;
    if lr_type == 'alex':
        if lr_owner == 'weight':
            return mx.AttrScope()
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='2.', wd_mult='0.')
        else:
            assert False
    # 10, 1; 20, 0;
    if lr_type == 'alex10':
        if lr_owner == 'weight':
            return mx.AttrScope(lr_mult='10.', wd_mult='1.')
        elif lr_owner == 'bias':
            return mx.AttrScope(lr_mult='20.', wd_mult='0.')
        else:
            assert False
    # 0, 0; 0, 0;
    # so apply this to both
    if lr_type == 'fixed':
        assert lr_owner in ('weight', 'bias')
        return mx.AttrScope(lr_mult='0.', wd_mult='0.')
    # 1, 1; 1, 1;
    # so do nothing
    return mx.AttrScope()


def layer(op):
    """
        Decorator for composable network layers.
    """
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        layer_name.append(name)
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, num_classes, filter_scale, evaluation=False, trainable=True, is_training=False):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.is_training = is_training
        self.trainable = trainable

        # Switch variable for dropout
        self.use_dropout = mx.sym.Variable(name='use_dropout')
        self.evaluation = evaluation
        self.filter_scale = filter_scale

        self.setup(is_training, num_classes, evaluation)

    def setup(self, is_training, num_classes, evaluation):
        """
        Construct the network. 
        """
        raise NotImplementedError('Must be implemented by the subclass.')

    @staticmethod
    def load(data_path, session, ignore_missing=False):
        """
        Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        """
        data_dict = np.load(data_path, encoding='latin1').item()
        print(data_dict)

    def feed(self, *args):
        """Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        """
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """
        Returns the current network output.
        """
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        """
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """Creates a new TensorFlow variable."""
        return mx.sym.Variable(name=name, shape=shape)

    def get_layer_name(self):
        return layer_name

    @staticmethod
    def validate_padding(padding):
        """Verifies that the padding is one of the supported ones."""
        assert padding in ('SAME', 'VALID')

    @layer
    def zero_padding(self, input, paddings, name):
        # pad_mat = np.array([[0,0], [paddings, paddings], [paddings, paddings], [0, 0]])
        return mx.sym.pad(data=input, mode="constant", constant_value=0, pad_width=(0,0,0,0,paddings,paddings,0,0), name=name)

    @layer
    def convolution(self, data, name, filters, kernel=3, stride=1, dilate=1, pad=-1,
             groups=1, no_bias=False, workspace=-1, cudnn_off=False, relu=True, padding=DEFAULT_PADDING):
        """
            convolution wrapper
        """
        if kernel == 1:
            # set dilate to 1, since kernel is 1
            dilate = 1
        if pad < 0:
            assert kernel % 2 == 1, 'Specify pad for an even kernel size'
            pad = ((kernel - 1) * dilate + 1) // 2
        if workspace < 0:
            workspace = cfg.get('workspace', 512)
        lr_type = cfg.get('lr_type', 'torch')
        with _attr_scope_lr(lr_type, 'weight'):
            weight = mx.sym.Variable('{}_weight'.format(name))
        if no_bias:
            output = mx.sym.Convolution(data=data, weight=weight, name=name,
                                      kernel=(kernel, kernel),
                                      stride=(stride, stride),
                                      dilate=(dilate, dilate),
                                      pad=(pad, pad),
                                      num_filter=filters,
                                      num_group=groups,
                                      workspace=workspace,
                                      no_bias=True,
                                      cudnn_off=cudnn_off)
        else:
            with _attr_scope_lr(lr_type, 'bias'):
                bias = mx.sym.Variable('{}_bias'.format(name))
            output = mx.sym.Convolution(data=data, weight=weight, bias=bias, name=name,
                                      kernel=(kernel, kernel),
                                      stride=(stride, stride),
                                      dilate=(dilate, dilate),
                                      pad=(pad, pad),
                                      num_filter=filters,
                                      num_group=groups,
                                      workspace=workspace,
                                      no_bias=False,
                                      cudnn_off=cudnn_off)

        if relu:
            output = mx.symbol.relu(data=output, name=name)

        return output

    @layer
    def pool(self, data, name, kernel=3, stride=2, dilate=1, pad=-1, pool_type='max', global_pool=False):
        """
            pooling wrapper
        """
        if isinstance(kernel, tuple):
            return mx.sym.Pooling(data, name=name,
                                  kernel=kernel,
                                  stride=stride,
                                  pool_type=pool_type)
        else:
            if pool_type == 'max+avg':
                branch1 = self.pool(data, '{}_branch1'.format(name),
                                    kernel=kernel,
                                    stride=stride,
                                    dilate=dilate,
                                    pad=pad,
                                    pool_type='max')
                branch2 = self.pool(data, '{}_branch2'.format(name),
                                    kernel=kernel,
                                    stride=stride,
                                    dilate=dilate,
                                    pad=pad,
                                    pool_type='avg')
                return branch1 + branch2
            if kernel == 1:
                assert dilate == 1
            if global_pool:
                assert dilate == 1
                assert pad < 0
                return mx.sym.Pooling(data, name=name,
                                      kernel=(1, 1),
                                      pool_type=pool_type,
                                      global_pool=True)
            else:
                if pad < 0:
                    if cfg.get('pool_top_infer_style', None) == 'caffe':
                        pad = 0
                    else:
                        assert kernel % 2 == 1, 'Specify pad for an even kernel size'
                        pad = ((kernel - 1) * dilate + 1) // 2
                if dilate == 1:
                    return mx.sym.Pooling(data, name=name,
                                          kernel=(kernel, kernel),
                                          stride=(stride, stride),
                                          pad=(pad, pad),
                                          pool_type=pool_type)
                else:
                    # TODO: not checked for stride > 1
                    assert stride == 1
                    return mx.sym.Pooling(data, name=name,
                                          kernel=(kernel, kernel),
                                          stride=(stride, stride),
                                          dilate=(dilate, dilate),
                                          pad=(pad, pad),
                                          pool_type=pool_type)

    @layer
    def fc(self, data, name, hiddens, no_bias=False):
        """
            fully connection wrapper
        """
        lr_type = cfg.get('lr_type', 'torch')
        with _attr_scope_lr(lr_type, 'weight'):
            weight = mx.sym.Variable('{}_weight'.format(name))
        if no_bias:
            return mx.sym.FullyConnected(data=data, weight=weight, name=name,
                                         num_hidden=hiddens,
                                         no_bias=True)
        else:
            with _attr_scope_lr(lr_type, 'bias'):
                bias = mx.sym.Variable('{}_bias'.format(name))
            return mx.sym.FullyConnected(
                data=data,
                weight=weight,
                bias=bias,
                name=name,
                num_hidden=hiddens,
                no_bias=False)

    @layer
    def softmax_out(self, data, grad_scale=1.0, multi_output=False, name='softmax'):
        """
            softmaxout wrapper
        """
        if multi_output:
            return mx.sym.SoftmaxOutput(data, name=name,
                                        grad_scale=grad_scale,
                                        use_ignore=True,
                                        ignore_label=255,
                                        multi_output=True,
                                        normalization='valid')
        else:
            return mx.sym.SoftmaxOutput(data, name=name,
                                        grad_scale=grad_scale,
                                        multi_output=False)

    @layer
    def relu(self, input, name):
        return mx.sym.Activation(data=input, name=name, act_type='relu')

    @layer
    def lrelu(self, data, name, slope=0.25):
        """
            lrelu wrapper
        """
        return mx.sym.LeakyReLU(data, name=name, act_type='leaky', slope=slope)

    @layer
    def bn(self, data, name, eps=1.001e-5, fix_gamma=False, use_global_stats=None, cudnn_off=False, relu=False):
        """
            batch normalization wrapper
        """
        if use_global_stats is None:
            use_global_stats = cfg.get('bn_use_global_stats', False)

        if fix_gamma:
            with mx.AttrScope(lr_mult='0.', wd_mult='0.'):
                gamma = mx.sym.Variable('{}_gamma'.format(name))
                beta = mx.sym.Variable('{}_beta'.format(name))
            output = mx.sym.BatchNorm(data=data, gamma=gamma, beta=beta, name=name,
                                    eps=eps,
                                    fix_gamma=True,
                                    use_global_stats=use_global_stats,
                                    cudnn_off=cudnn_off)
        else:
            lr_type = cfg.get('lr_type', 'torch')
            with _attr_scope_lr(lr_type, 'weight'):
                gamma = mx.sym.Variable('{}_gamma'.format(name))
            with _attr_scope_lr(lr_type, 'bias'):
                beta = mx.sym.Variable('{}_beta'.format(name))
            output = mx.sym.BatchNorm(data=data, gamma=gamma, beta=beta, name=name,
                                    eps=eps,
                                    fix_gamma=False,
                                    use_global_stats=use_global_stats)

        if relu:
            output = mx.sym.relu(output)

        return output

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        #
        # nsize = 2 * depth_radius
        # knorm = bias
        return mx.sym.LRN(data=input,
                          alpha=alpha,
                          beta=beta,
                          knorm=bias,
                          nsize=2*radius,
                          name=name)

    @layer
    def concat(self, inputs, axis, name):
        return mx.sym.concat(inputs, axis, name=name)

    @layer
    def add(self, inputs, name):
        return mx.sym.ElementWiseSum(*inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True):
        arg_shape, output_shape, aux_shape = input.infer_shape()
        input_shape = input.get_shape()
        if input_shape.ndims == 4:
            # The input is spatial. Vectorize it first.
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = mx.sym.reshape(input, [-1, dim])
        else:
            feed_in, dim = (input, input_shape[-1].value)
        weights = self.make_var('weights', shape=[dim, num_out])
        biases = self.make_var('biases', [num_out])
        fc = mx.sym.FullyConnected(data=feed_in,
                                   weight=weights,
                                   bias=biases,
                                   name=name)
        if relu:
            fc = mx.sym.relu(data=fc,
                             name=name)
        return fc

    @layer
    def lrelu(data, name, slope=0.25):
        """
            lrelu wrapper
        """
        return mx.sym.LeakyReLU(data, name=name, act_type='leaky', slope=slope)

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return mx.sym.Dropout(data=input, p=keep, name=name)

    @layer
    def resize_bilinear(self, input, size, name):
        arg_shape, output_shape, aux_shape = input.infer_shape()
        shape = list(output_shape[0])
        size = list(size)
        shape[2] = size[0]
        shape[3] = size[1]
        # shape = tuple(shape)
        # output = mx.sym.Variable(name=name, shape=shape)
        return mx.image.imresize(src=input,
                                 h=size[0],
                                 w=size[1],
                                 name=name)

    @layer
    def up_sample(self, input, scale=1, name=None):
        output = mx.sym.UpSampling(data=input, scale=int(scale), sample_type='bilinear', name=name)
        return output

    @layer
    def interp(self, input, s_factor=1, z_factor=1, name=None):
        # ori_h, ori_w = input.get_shape().as_list()[1:3]
        arg_shape, output_shape, aux_shape = input.infer_shape()
        ori_h, ori_w = output_shape[0][2:]

        # shrink
        ori_h = (ori_h - 1) * s_factor + 1
        ori_w = (ori_w - 1) * s_factor + 1
        # zoom
        ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
        ori_w = ori_w + (ori_w - 1) * (z_factor - 1)
        resize_shape = [int(ori_h), int(ori_w)]

        output = mx.image.imresize(src=input,
                                 h=resize_shape[0],
                                 w=resize_shape[1],
                                 name=name)
        return output
