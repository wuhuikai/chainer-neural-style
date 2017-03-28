import collections

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, link, function, initializers

def gram(x):
    b, c, h, w = x.shape
    feature = F.reshape(x, (b, c, h*w))
    return F.batch_matmul(feature, feature, transb=True) / (c*h*w)

class NormalizeGrad(function.Function):
    def __init__(self, func):
        self._eps = 10e-8
        self._func = func

    def forward(self, inputs):
        return self._func.forward(inputs)

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        grads = self._func.backward(inputs, grad_outputs)

        return tuple(grad/(xp.linalg.norm(grad.flatten(), ord=1)+self._eps) for grad in grads)

def normlize_grad(func, inputs, normalize=True):
    return NormalizeGrad(func)(*inputs) if normalize else func(*inputs)

def extract(inputs, model, layers, train=False):
    features = {}
    layers = set(layers)
    variables = dict(inputs)

    model.train = train
    for func_name, bottom, top in model.layers:
        if len(layers) == 0:
            break
        if func_name not in model.forwards or any(blob not in variables for blob in bottom):
            continue
        func = model.forwards[func_name]
        input_vars = tuple(variables[blob] for blob in bottom)
        output_vars = func(*input_vars)
        if not isinstance(output_vars, collections.Iterable):
            output_vars = output_vars,
        for var, name in zip(output_vars, top):
            variables[name] = var
        if func_name in layers:
            features[func_name] = output_vars
            layers.remove(func_name)
    return features

def total_variation(x):
    _, _, h, w = x.data.shape

    return 0.5*F.sum((x[:, :, :h-1, :w-1] - x[:, :, 1:, :w-1])**2 + (x[:, :, :h-1, :w-1] - x[:, :, :h-1, 1:])**2)

class InstanceNormalization(link.Link):
    def __init__(self, nc, dtype=np.float32):
        super(InstanceNormalization, self).__init__()
        self.nc = nc
        self.dtype = dtype
        self.bn = None
        self.prev_batch = None

        self.add_param('gamma', nc, dtype=dtype)
        initializers.init_weight(self.gamma.data, np.random.uniform(size=nc))

        self.add_param('beta', nc, dtype=dtype)
        initializers.init_weight(self.beta.data, initializers.Zero())

    def __call__(self, x, test=True):
        n, c, h, w = x.shape
        assert(c == self.nc)
        if n != self.prev_batch:
            self.bn = L.BatchNormalization(n*c, dtype=self.dtype)
            self.bn.to_gpu(self._device_id)
            self.bn.gamma = F.tile(self.gamma, n)
            self.bn.beta = F.tile(self.beta, n)
            self.prev_batch = n

        x = F.reshape(x, (1, n*c, h, w))
        return F.reshape(self.bn(x), (n, c, h, w))

class ResidualBlock(chainer.Chain):
    def __init__(self, nc, stride=1, ksize=3, instance_normalization=True, w_init=None):
        BN = InstanceNormalization if instance_normalization else L.BatchNormalization
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(None, nc, ksize=ksize, stride=stride, pad=1, initialW=w_init),
            c2=L.Convolution2D(None, nc, ksize=ksize, stride=stride, pad=1, initialW=w_init),
            b1=BN(nc),
            b2=BN(nc)
        )

    def __call__(self, x, test):
        h = F.relu(self.b1(self.c1(x), test=test))
        h = self.b2(self.c2(h), test=test)
        
        return h + x

class ImageTransformer(chainer.Chain):
    def __init__(self, feature_map_nc, output_nc, tanh_constant, instance_normalization=True, w_init=None):
        self.tanh_constant = tanh_constant
        BN = InstanceNormalization if instance_normalization else L.BatchNormalization
        super(ImageTransformer, self).__init__(
            c1=L.Convolution2D(None, feature_map_nc, ksize=9, stride=1, pad=4, initialW=w_init),
            c2=L.Convolution2D(None, 2*feature_map_nc, ksize=3, stride=2, pad=1, initialW=w_init),
            c3=L.Convolution2D(None, 4*feature_map_nc, ksize=3,stride=2, pad=1, initialW=w_init),
            r1=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r2=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r3=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r4=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r5=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            d1=L.Deconvolution2D(None, 2*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            d2=L.Deconvolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            d3=L.Convolution2D(None, output_nc, ksize=9, stride=1, pad=4, initialW=w_init),
            b1=BN(feature_map_nc),
            b2=BN(2*feature_map_nc),
            b3=BN(4*feature_map_nc),
            b4=BN(2*feature_map_nc),
            b5=BN(feature_map_nc)
        )

    def __call__(self, x, test=False):
        h = F.relu(self.b1(self.c1(x), test=test))
        h = F.relu(self.b2(self.c2(h), test=test))
        h = F.relu(self.b3(self.c3(h), test=test))
        h = self.r1(h, test=test)
        h = self.r2(h, test=test)
        h = self.r3(h, test=test)
        h = self.r4(h, test=test)
        h = self.r5(h, test=test)
        h = F.relu(self.b4(self.d1(h), test=test))
        h = F.relu(self.b5(self.d2(h), test=test))
        y = self.d3(h)

        return F.tanh(y) * self.tanh_constant