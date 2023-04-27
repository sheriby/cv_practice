import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.vgg import vgg19

from nets.srgan import Discriminator, Generator
from utils.callbacks import LossHistory
from utils.dataloader import SRGAN_dataset_collate, SRGANDataset
from utils.utils import (download_weights, get_lr_scheduler, set_optimizer_lr,
                         show_config)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    Cuda            = True
    distributed     = False
    fp16            = False
    G_model_path    = ""
    D_model_path    = ""
    scale_factor    = 4
    lr_shape        = [96, 96]
    hr_shape        = [lr_shape[0] * scale_factor, lr_shape[1] * scale_factor]
    Init_Epoch      = 0
    Epoch           = 200
    batch_size      = 4
    Init_lr         = 2e-4
    Min_lr          = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.9
    weight_decay        = 0
    lr_decay_type       = "cos"
    save_period         = 10
    save_dir            = 'logs'
    num_workers         = 4
    photo_save_step     = 50
    annotation_path = "train_lines.txt"
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0

    if distributed:
        if local_rank == 0:
            download_weights()  
        dist.barrier()
    else:
        download_weights()

    G_model = Generator(scale_factor)
    D_model = Discriminator()
    
    VGG_model = vgg19(pretrained=True)
    VGG_feature_model = nn.Sequential(*list(VGG_model.features)[:-1]).eval()
    for param in VGG_feature_model.parameters():
        param.requires_grad = False
    if G_model_path != '':
        model_dict      = G_model.state_dict()
        pretrained_dict = torch.load(G_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        G_model.load_state_dict(model_dict)
    if D_model_path != '':
        model_dict      = D_model.state_dict()
        pretrained_dict = torch.load(D_model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        D_model.load_state_dict(model_dict)

    BCE_loss = nn.BCEWithLogitsLoss()
    MSE_loss = nn.MSELoss()
    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, [G_model], input_shape=lr_shape)
    else:
        loss_history    = None
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    G_model_train = G_model.train()
    D_model_train = D_model.train()
    
    if Cuda:
        if distributed:
            G_model_train = G_model_train.cuda(local_rank)
            G_model_train = torch.nn.parallel.DistributedDataParallel(G_model_train, device_ids=[local_rank], find_unused_parameters=True)
            
            D_model_train = D_model_train.cuda(local_rank)
            D_model_train = torch.nn.parallel.DistributedDataParallel(D_model_train, device_ids=[local_rank], find_unused_parameters=True)
            
            VGG_feature_model = VGG_feature_model.cuda(local_rank)
        else:
            cudnn.benchmark = True
            G_model_train = torch.nn.DataParallel(G_model)
            G_model_train = G_model_train.cuda()

            D_model_train = torch.nn.DataParallel(D_model)
            D_model_train = D_model_train.cuda()    
            
            VGG_feature_model = torch.nn.DataParallel(VGG_feature_model)
            VGG_feature_model = VGG_feature_model.cuda()

    with open(annotation_path) as f:
        lines = f.readlines()
    num_train = len(lines)

    if local_rank == 0:
        show_config(
            lr_shape = lr_shape, hr_shape = hr_shape, Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train
            )

    if True:
        G_optimizer = {
            'adam'  : optim.Adam(G_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(G_model_train.parameters(), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        D_optimizer = {
            'adam'  : optim.Adam(D_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(D_model_train.parameters(), Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Epoch)
        
        epoch_step      = min(num_train // batch_size, 2000)
        if epoch_step == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        train_dataset   = SRGANDataset(lines, lr_shape, hr_shape)
        
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            shuffle         = True
    
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=SRGAN_dataset_collate, sampler=train_sampler)

        for epoch in range(Init_Epoch, Epoch):

            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(G_optimizer, lr_scheduler_func, epoch)
            set_optimizer_lr(D_optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(G_model_train, D_model_train, G_model, D_model, VGG_feature_model, loss_history, G_optimizer, D_optimizer, BCE_loss, MSE_loss, 
                        epoch, epoch_step, gen, Epoch, Cuda, fp16, scaler, save_period, save_dir, photo_save_step, local_rank)

            if distributed:
                dist.barrier()
