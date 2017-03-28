from __future__ import print_function
import os
import argparse

from chainer.links import caffe

import pickle

def main():
    parser = argparse.ArgumentParser(description='Load caffe model for chainer')
    parser.add_argument('--caffe_model_path', default='models/VGG_ILSVRC_19_layers.caffemodel', help='Path for caffe model')
    args = parser.parse_args()

    print('Load caffe model from {} ...'.format(args.caffe_model_path))
    caffe_model = caffe.CaffeFunction(args.caffe_model_path)
    print('Load caffe model, DONE')

    save_path = '{}.pkl'.format(os.path.splitext(args.caffe_model_path)[0])
    print('\nSave to {} ...'.format(save_path))
    with open(save_path, 'wb') as f:
        pickle.dump(caffe_model, f)

if __name__ == '__main__':
    main()