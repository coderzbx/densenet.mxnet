
import mxnet as mx
from symbol_densenet import DenseNet

num_classes = 15
units = [6, 12, 32, 32]
densenet = DenseNet(units=units, num_stage=4, growth_rate=32, num_class=num_classes,
                    data_type="kd", reduction=0.5, drop_out=0.2, bottle_neck=True,
                    bn_mom=0.90, workspace=512)

dot = mx.viz.plot_network(densenet)
dot.render('densenet')

