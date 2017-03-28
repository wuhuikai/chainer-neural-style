from __future__ import print_function
import os
import pickle
import argparse

import chainer
from chainer import training
from chainer.training import extensions

from model import ImageTransformer
from updater import StyleUpdater, display_image
from dataset import SuperImageDataset

str2list = lambda x: x.split(';')
str2bool = lambda x:x.lower() == 'true'

def make_optimizer(model, alpha):
    optimizer = chainer.optimizers.Adam(alpha=alpha)
    optimizer.setup(model)
    return optimizer

def main():
    parser = argparse.ArgumentParser(description='Fast neural style transfer')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--filter_num', type=int, default=32, help="# of filters in ImageTransformer's 1st conv layer")
    parser.add_argument('--output_channel', type=int, default=3, help='# of output image channels')
    parser.add_argument('--tanh_constant', type=float, default=150, help='Constant for output of ImageTransformer')
    parser.add_argument('--instance_normalization', type=str2bool, default=True, help='Use InstanceNormalization if True')
    parser.add_argument('--model_path', default='models/VGG_ILSVRC_19_layers.pkl', help='Path for pretrained model')

    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for Adam')
    parser.add_argument('--n_iterations', type=int, default=40000, help='# of iterations for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of images in each mini-batch')
    parser.add_argument('--n_thread', type=int, default=10, help='# of workers for loading data')
    parser.add_argument('--load_size', type=int, default=256, help='Scale image to load_size')
    parser.add_argument('--out', default='fast_style_result', help='Directory to output the result')
    
    ## Different layers & model
    parser.add_argument('--content_layers', type=str2list, default='relu4_2', help='Layers for content_loss, sperated by ;')
    parser.add_argument('--content_weight', type=float, default=1, help='Weight for content loss')
    parser.add_argument('--style_layers', type=str2list, default='relu1_1;relu2_1;relu3_1;relu4_1;relu5_1', help='Layers for style_loss, sperated by ;')
    parser.add_argument('--style_weight', type=float, default=5, help='Weight for style loss')
    parser.add_argument('--tv_weight', type=float, default=1e-6, help='Weight for tv loss')
    parser.add_argument('--style_image_path', default='images/Starry_Night.jpg', help='Style src image')
    parser.add_argument('--style_load_size', type=int, default=256, help='Scale style image to load_size')

    parser.add_argument('--data_root', help='Path for dataset root folder')
    parser.add_argument('--train_folder', default='train2014', help='Folder for storing train images')
    parser.add_argument('--val_folder', default='val2014', help='Folder for storing val images')
    parser.add_argument('--train_list', default='train.txt', help='File storing train image list ')
    parser.add_argument('--val_list', default='val.txt', help='File storing val images list')

    parser.add_argument('--resume', default='', help='Resume the training from snapshot')    
    parser.add_argument('--snapshot_interval', type=int, default=1000, help='Interval of snapshot (iterations)')
    parser.add_argument('--print_interval', type=int, default=50, help='Interval of printing log to console (iteration)')
    parser.add_argument('--plot_interval', type=int, default=100, help='Interval of plot (iteration)')
    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    # Set up ImageTransformer & VGG
    print('Create & Init models ...')
    G = ImageTransformer(args.filter_num, args.output_channel, args.tanh_constant, args.instance_normalization)
    with open(args.model_path, 'rb') as f:
        D = pickle.load(f)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        G.to_gpu()                               # Copy the model to the GPU
        D.to_gpu()
    print('Init models done ...\n')

    # Setup an optimizer
    optimizer = make_optimizer(G, args.lr)

    ########################################################################################################################
    # Setup dataset & iterator
    trainset = SuperImageDataset(os.path.join(args.data_root, args.train_list), root=os.path.join(args.data_root, args.train_folder), load_size=args.load_size)
    valset = SuperImageDataset(os.path.join(args.data_root, args.val_list), root=os.path.join(args.data_root, args.val_folder))
    print('Trainset contains {} image files'.format(len(trainset)))
    print('Valset contains {} image files'.format(len(valset)))
    print('')
    train_iter = chainer.iterators.MultiprocessIterator(trainset, args.batch_size, n_processes=args.n_thread, n_prefetch=args.n_thread)
    val_iter = chainer.iterators.MultiprocessIterator(trainset, args.batch_size, n_processes=args.n_thread, n_prefetch=args.n_thread)
    ########################################################################################################################

    # Set up a trainer
    updater = StyleUpdater(
        models=(G, D),
        args=args,
        iterator=train_iter,
        optimizer=optimizer,
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.n_iterations, 'iteration'), out=args.out)

    # Snapshot
    snapshot_interval = (args.snapshot_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        G, 'transformer_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    
    # Display
    print_interval = (args.print_interval, 'iteration')
    trainer.extend(extensions.LogReport(trigger=print_interval))
    trainer.extend(extensions.PrintReport([
        'iteration', 'main/loss', 'main/content_loss', 'main/style_loss', 'main/tv_loss'
    ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=args.print_interval))

    trainer.extend(extensions.dump_graph('main/loss', out_name='TrainGraph.dot'))

    # Plot
    plot_interval = (args.plot_interval, 'iteration')

    trainer.extend(
        extensions.PlotReport(['main/loss'], 'iteration', file_name='loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(
        extensions.PlotReport(['main/content_loss'], 'iteration', file_name='content_loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(
        extensions.PlotReport(['main/style_loss'], 'iteration', file_name='style_loss.png', trigger=plot_interval), trigger=plot_interval)
    trainer.extend(
        extensions.PlotReport(['main/tv_loss'], 'iteration', file_name='tv_loss.png', trigger=plot_interval), trigger=plot_interval)

    # Eval
    trainer.extend(display_image(G, valset, os.path.join(args.out, 'val'), args.gpu), trigger=plot_interval)
    
    if args.resume:
        # Resume from a snapshot
        print('Resume from {} ... \n'.format(args.resume))
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    print('Training start ...\n')
    trainer.run()

if __name__ == '__main__':
    main()