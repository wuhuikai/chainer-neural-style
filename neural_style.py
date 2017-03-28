from __future__ import print_function
import os
import argparse

import chainer
import chainer.functions as F
from chainer import Variable, cuda

import pickle
import numpy as np
from scipy.optimize import minimize
from skimage.io import imread, imsave
from skimage.color import rgb2yuv, yuv2rgb

from model import total_variation, extract, normlize_grad, gram
from utils import im_preprocess_vgg, im_deprocess_vgg

str2list = lambda x: x.split(';')
str2bool = lambda x:x.lower() == 'true'

def shape_color(shape, color):
    y = rgb2yuv(shape)[:,:,:1]
    uv = rgb2yuv(color)[:,:,1:]
    return np.clip(yuv2rgb(np.concatenate((y, uv), axis=2)) * 255, 0, 255)

def style_colors(content, img):
    return shape_color(content, img)

def original_colors(content, img):
    return shape_color(img, content)

def neural_style(x, model, content_features, grams, args):
    # TV loss
    x = np.asarray(np.reshape(x, args.shape), dtype=np.float32)
    x = Variable(chainer.dataset.concat_examples([x], args.gpu))
    loss = args.tv_weight * total_variation(x)

    # Extract features for x
    layers = args.content_layers | args.style_layers
    x_features = extract({'data': x}, model, layers)
    x_features = {key:value[0] for key, value in x_features.items()}
    
    # Concent loss
    for layer in args.content_layers:
        loss += args.content_weight * normlize_grad(F.MeanSquaredError(), (content_features[layer], x_features[layer]), normalize=args.normalize_gradients)
    
    # Style loss
    for layer in args.style_layers:
        loss += args.style_weight * normlize_grad(F.MeanSquaredError(), (grams[layer], gram(x_features[layer])), normalize=args.normalize_gradients)
    
    loss.backward()

    # GPU to CPU
    loss = cuda.to_cpu(loss.data)
    diff = np.asarray(cuda.to_cpu(x.grad).flatten(), dtype=np.float64)

    return loss, diff

def main():
    parser = argparse.ArgumentParser(description='Transfer style from src image to target image')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--content_image', default='images/towernight.jpg', help='Content target image')
    parser.add_argument('--style_images', type=str2list, default='images/Starry_Night.jpg', help='Style src images, sperated by ;')
    parser.add_argument('--blend_weights', type=lambda x: np.array([float(i) for i in x.split(';')]), default=None, help='Weight for each style image, sperated by ;')

    parser.add_argument('--content_weight', type=float, default=5, help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=100, help='Weight for style loss')
    parser.add_argument('--tv_weight', type=float, default=1e-3, help='Weight for tv loss')
    parser.add_argument('--n_iteration', type=int, default=1000, help='# of iterations')
    parser.add_argument('--normalize_gradients', type=str2bool, default=False, help='Normalize gradients if True')
    parser.add_argument('--rand_init', type=str2bool, default=True, help='Random init input if True')
    parser.add_argument('--content_load_size', type=int, default=512, help='Scale content image to load_size')
    parser.add_argument('--style_load_size', type=int, default=512, help='Scale style image to load_size')
    parser.add_argument('--original_color', type=str2bool, default=False, help='Same color with content image if True')
    parser.add_argument('--style_color', type=str2bool, default=False, help='Same color with style image if True')

    parser.add_argument('--content_layers', type=str2list, default='relu4_2', help='Layers for content_loss, sperated by ;')
    parser.add_argument('--style_layers', type=str2list, default='relu1_1;relu2_1;relu3_1;relu4_1;relu5_1', help='Layers for style_loss, sperated by ;')

    parser.add_argument('--model_path', default='models/VGG_ILSVRC_19_layers.pkl', help='Path for pretrained model')
    parser.add_argument('--out_folder', default='images/result', help='Folder for storing output result')
    parser.add_argument('--prefix', default='', help='Prefix name for output image')
    args = parser.parse_args()

    print('Load pretrained model from {} ...'.format(args.model_path))
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()    # Make a specified GPU current
        model.to_gpu()                             # Copy the model to the GPU

    print('Load content image {} ...'.format(args.content_image))
    content_im_orig = imread(args.content_image)
    args.content_orig_size = content_im_orig.shape[:2] if args.content_load_size else None
    content_im = im_preprocess_vgg(content_im_orig, load_size=args.content_load_size, dtype=np.float32)
    args.shape = content_im.shape
    print('Load style image(s) ...\n\t{}'.format('\t'.join(args.style_images)))
    style_images = [im_preprocess_vgg(imread(im_path), load_size=args.style_load_size, dtype=np.float32) for im_path in args.style_images]

    if args.blend_weights is None:
        args.blend_weights = np.ones(len(style_images))
    args.blend_weights /= np.sum(args.blend_weights)
    print('Blending weight for each stype image: {}'.format(args.blend_weights))

    # Init x
    x = np.asarray(np.random.randn(*content_im.shape) * 0.001, dtype=np.float32) if args.rand_init else np.copy(content_im)

    print('Extracting content image features ...')
    args.content_layers = set(args.content_layers)
    content_im = Variable(chainer.dataset.concat_examples([content_im], args.gpu), volatile='on')
    content_features = extract({'data': content_im}, model, args.content_layers)
    content_features = {key:value[0] for key, value in content_features.items()}
    for _, value in content_features.items():
        value.volatile = 'off'

    print('Extracting style image features ...')
    grams = {}
    args.style_layers = set(args.style_layers)
    for i, style_image in enumerate(style_images):
        style_image = Variable(chainer.dataset.concat_examples([style_image], args.gpu), volatile='on')
        style_features = extract({'data': style_image}, model, args.style_layers)
        for key, value in style_features.items():
            gram_feature = gram(value[0])
            if key in grams:
                grams[key] += args.blend_weights[i]*gram_feature
            else:
                grams[key] = args.blend_weights[i]*gram_feature
    for _, value in grams.items():
        value.volatile = 'off'

    print('Optimize start ...')
    res = minimize(neural_style, x, args=(model, content_features, grams, args), method='L-BFGS-B', jac=True, options={'maxiter': args.n_iteration, 'disp':True})
    loss0, _ = neural_style(x, model, content_features, grams, args)

    print('Optimize done, loss = {}, with loss0 = {}'.format(res.fun, loss0))
    img = im_deprocess_vgg(np.reshape(res.x, args.shape), orig_size=args.content_orig_size, dtype=np.uint8)
    if args.original_color:
        img = original_colors(content_im_orig, img)
    if args.style_color:
        img = style_colors(content_im_orig, img)
    img = np.asarray(img, dtype=np.uint8)

    # Init result list
    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)
    print('Result will save to {} ...\n'.format(args.out_folder))

    name = '{}_with_style(s)'.format(os.path.splitext(os.path.basename(args.content_image))[0])
    for path in args.style_images:
        name = '{}_{}'.format(name, os.path.splitext(os.path.basename(path))[0])
    if args.prefix:
        name = '{}_{}'.format(args.prefix, name)
    imsave(os.path.join(args.out_folder, '{}.png'.format(name)), img)

if __name__ == '__main__':
    main()