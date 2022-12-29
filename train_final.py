import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
#import torch.utils.data.distributed
from models.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import datetime
from dataloader.dataset_NIA import train_data_loader, test_data_loader
from runner_helper import *


class Pseudoarg():
    def __init__(self):
        self.workers = 1
        self.epochs = 100
        self.start_epoch = 0
        self.batch_size = 64
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 10
        self.resume = None
        self.data_set = 10
        
args = Pseudoarg()

# %%
now = datetime.datetime.now()
time_str = now.strftime("%m%d_%H%M_")
project_path = "./"
log_txt_path = project_path + 'log.txt'
log_curve_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'


# %%
#def main():
best_acc = 0
recorder = RecorderMeter(args.epochs)
print('The training time: ' + now.strftime("%m-%d %H:%M"))
print('The training set: set ' + str(args.data_set))
os.makedirs(project_path+"log/",exist_ok = True)
with open(log_txt_path, 'a') as f:
    f.write('The training set: set ' + str(args.data_set) + '\n')

# create model and load pre_trained parameters
model = GenerateModel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.nn.DataParallel(model).cuda()
print(model)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


# %%
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        recorder = checkpoint['recorder']
        best_acc = best_acc.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
cudnn.benchmark = True

# %%
# Data loading code
train_data = train_data_loader(project_dir=project_path, 
                               data_set=args.data_set)
test_data = test_data_loader(project_dir=project_path,
                             data_set=args.data_set)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers,
                                           pin_memory=True,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(test_data,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=True)

for epoch in range(args.start_epoch, args.epochs):
    inf = '********************' + str(epoch) + '********************'
    start_time = time.time()
    current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

    with open(log_txt_path, 'a') as f:
        f.write(inf + '\n')
        f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

    print(inf)
    print('Current learning rate: ', current_learning_rate)

    # train for one epoch
    train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, log_txt_path, args)

    # evaluate on validation set
    val_acc, val_los = validate(val_loader, model, criterion, log_txt_path, args)

    scheduler.step()

    # remember best acc and save checkpoint
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)
    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(),
                     'recorder': recorder}, 
                     checkpoint_path, 
                     best_checkpoint_path,
                     is_best,
                     )

    # print and save log
    epoch_time = time.time() - start_time
    recorder.update(epoch, train_los, train_acc, val_los, val_acc)
    recorder.plot_curve(log_curve_path)

    print('The best accuracy: {:.3f}'.format(best_acc.item()))
    print('An epoch time: {:.1f}s'.format(epoch_time))
    with open(log_txt_path, 'a') as f:
        f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
        f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')

# %%



