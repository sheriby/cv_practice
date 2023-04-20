import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of dataset
#directory to save weights file
CHECKPOINT_PATH = 'checkpoint'

#total training epoches
# EPOCH = 200
# MILESTONES = [60, 120, 160]
EPOCH = 20
# MILESTONES = [5, 10, 15]
MILESTONES = [40, 60, 80]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10







