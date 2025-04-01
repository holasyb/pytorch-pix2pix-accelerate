"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

from accelerate import Accelerator, utils
import torch
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from accelerate import DistributedDataParallelKwargs
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    utils.set_seed(3407)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    opt.accelerator = accelerator

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    model.netG, model.optimizer_G, model.netD, model.optimizer_D, dataset.dataloader = accelerator.prepare(
        model.netG, model.optimizer_G, model.netD, model.optimizer_D, dataset.dataloader
    )
    def replace_batchnorm_with_syncbatchnorm(model):
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                sync_bn = nn.SyncBatchNorm(module.num_features, module.eps, module.momentum, module.affine, module.track_running_stats)
                if hasattr(module, 'weight') and module.weight is not None:
                    sync_bn.weight.data = module.weight.data.clone().detach()
                if hasattr(module, 'bias') and module.bias is not None:
                    sync_bn.bias.data = module.bias.data.clone().detach()
                if hasattr(module, 'running_mean'):
                    sync_bn.running_mean.data = module.running_mean.data.clone().detach()
                if hasattr(module, 'running_var'):
                    sync_bn.running_var.data = module.running_var.data.clone().detach()
                if hasattr(module, 'num_batches_tracked'):
                    sync_bn.num_batches_tracked.data = module.num_batches_tracked.data.clone().detach()

                parent_name = name.rsplit('.', 1)[0]
                if parent_name:
                    parent = model.get_submodule(parent_name)
                    module_name = name.split('.')[-1]
                    setattr(parent, module_name, sync_bn)
                elif name == '':
                    # 如果是顶层模型，直接替换 (理论上不应该发生)
                    return sync_bn
        return model

    model.netG = replace_batchnorm_with_syncbatchnorm(model.netG)
    model.netD = replace_batchnorm_with_syncbatchnorm(model.netD)

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing

            model.optimizer_G.zero_grad()
            model.optimizer_D.zero_grad()
            
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
