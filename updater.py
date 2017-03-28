import os

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from skimage.io import imread, imsave

from model import extract, gram, total_variation
from utils import im_preprocess_vgg, im_deprocess_vgg

def display_image(G, valset, dst, device):
    @chainer.training.make_extension()
    def make_image(trainer):
        preview_dir = os.path.join(dst, 'preview')
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        idx = np.random.randint(0, len(valset))
        img = valset.get_example(idx)
        input_var = Variable(chainer.dataset.concat_examples([img], device), volatile='on')

        out_var = G(input_var, test=True)
        out = np.squeeze(chainer.cuda.to_cpu(out_var.data))
        out_img = im_deprocess_vgg(out, dtype=np.uint8)

        name = valset.get_name(idx)
        imsave(os.path.join(preview_dir, name), out_img)

    return make_image

class StyleUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.args = kwargs.pop('args')
        self.args.content_layers = set(self.args.content_layers)
        self.args.style_layers = set(self.args.style_layers)
        self.layers = self.args.content_layers | self.args.style_layers
        
        print('Extract style feature from {} ...\n'.format(self.args.style_image_path))
        style_image = im_preprocess_vgg(imread(self.args.style_image_path), load_size=self.args.style_load_size, dtype=np.float32)
        style_image_var = Variable(chainer.dataset.concat_examples([style_image], self.args.gpu), volatile='on')
        style_features = extract({'data': style_image_var}, self.D, self.args.style_layers)
        self.grams = {}
        for key, value in style_features.items():
            gram_feature = gram(value[0])
            _, w, h = gram_feature.shape
            gram_feature = F.broadcast_to(gram_feature, (self.args.batch_size, w, h))
            gram_feature.volatile = 'off'
            self.grams[key] = gram_feature
    
        super(StyleUpdater, self).__init__(*args, **kwargs)

    def loss(self, ouput_features, content_features, output_var):
        content_loss = 0
        for layer in self.args.content_layers:
            content_loss += F.mean_squared_error(content_features[layer], ouput_features[layer][0])

        style_loss = 0
        for layer in self.args.style_layers:
            style_loss += F.mean_squared_error(self.grams[layer], gram(ouput_features[layer][0]))

        tv_loss = total_variation(output_var)

        loss = self.args.content_weight*content_loss + self.args.style_weight*style_loss + self.args.tv_weight*tv_loss
        chainer.report({'content_loss': content_loss, 'style_loss': style_loss, 'tv_loss': tv_loss, 'loss': loss}, self.G)

        return loss
    
    def update_core(self):
        batch = self.get_iterator('main').next()
        input_var = Variable(self.converter(batch, self.device), volatile='on')

        content_features = extract({'data': input_var}, self.D, self.args.content_layers)
        content_features = {key:value[0] for key, value in content_features.items()}
        for _, value in content_features.items():
            value.volatile = 'off'
        
        input_var.volatile = 'off'
        output_var = self.G(input_var)
        ouput_features = extract({'data': output_var}, self.D, self.layers)
    
        optimizer = self.get_optimizer('main')
        optimizer.update(self.loss, ouput_features, content_features, output_var)