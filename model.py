import collections

import chainer.functions as F

from chainer import cuda, function

def gram(x):
    b, c, h, w = x.shape
    feature = F.reshape(x, (b, c, h*w))
    return F.batch_matmul(feature, feature, transb=True) / (c*h*w)

def normlize_grad(func, inputs, normalize=True):
    return NormalizeGrad(func)(*inputs) if normalize else func(*inputs)

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