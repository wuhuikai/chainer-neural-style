import os
import numpy
from skimage.io import imread

from chainer.dataset import dataset_mixin

from utils import im_preprocess_vgg

class SuperImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', load_size=None, crop_size=None, flip=False, dtype=numpy.float32):
        with open(paths) as paths_file:
            self._paths = [path.strip() for path in paths_file]
        self._root = root
        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size
        self._flip = flip

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return os.path.basename(self._paths[i])

    def get_example(self, i):
        img = im_preprocess_vgg(imread(os.path.join(self._root, self._paths[i])), load_size=self._load_size, dtype=self._dtype)
        if self._crop_size:
            _, w, h = img.shape
            sx, sy = numpy.random.randint(0, w - self._crop_size), numpy.random.randint(0, h - self._crop_size)
            img = img[:, sx:sx+self._crop_size, sy:sy+self._crop_size]
        if self._flip and numpy.random.rand() > 0.5:
            img = img[:, :, ::-1]
        
        return img