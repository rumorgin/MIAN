import argparse
import os
import time

import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Models.dataloader.samplers import CategoriesSampler

from Models.models.Protonet import Proto
from Models.utils import *
from Models.dataloader.data_utils import *

DATA_DIR='/share/qinzhili/New_Model/datasets'

parser = argparse.ArgumentParser()
# about dataset and network
parser.add_argument('-dataset', type=str, default='tieredimagenet', choices=['miniimagenet', 'cub','tieredimagenet','fc100','tieredimagenet_yao','cifar_fs'])
parser.add_argument('-data_dir', type=str, default=DATA_DIR)
parser.add_argument('-patch_list',type=str,default='3',help='the size of grids at every image-pyramid level')
parser.add_argument('-patch_ratio',type=float,default=2,help='scale the patch to incorporate context around the patch')
# about pre-training
parser.add_argument('-max_epoch', type=int, default=120)
parser.add_argument('-lr', type=float, default=0.05)
parser.add_argument('-step_size', type=int, default=30)
parser.add_argument('-gamma', type=float, default=0.2)
parser.add_argument('-bs', type=int, default=48)
parser.add_argument('-dropout',type=float, default=0)
# about validation
parser.add_argument('-set', type=str, default='val', choices=['val', 'test'], help='the set for validation')
parser.add_argument('-way', type=int, default=5)
parser.add_argument('-shot', type=int, default=1)
parser.add_argument('-query', type=int, default=5)

parser.add_argument('-num_episode', type=int, default=100)
parser.add_argument('-save_all', action='store_true', help='save models on each epoch')
parser.add_argument('-random_val_task',type=bool,default=True,help='random samples tasks for validation at each epoch')
# about deepemd setting
parser.add_argument('-deepemd', type=str, default='grid', choices=['fcn', 'grid', 'sampling'])
parser.add_argument('-feature_pyramid', type=str, default=None)
parser.add_argument('-num_patch',type=int,default=10)
parser.add_argument('-num_attention',type=int,default=1,help='the number of the attention generated proto vector ')
# about training
parser.add_argument('-gpu', default='0')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-extra_dir', type=str,default=None,help='extra information that is added to checkpoint dir, e.g. hyperparameters')
parser.add_argument('-num_workers',type=int,default=24,help='the number of dataloader workers')
parser.add_argument('-use_gpu',type=bool,default=True,help='if the code running with gpu')
parser.add_argument('-pretrain',type=bool,default=True,help='if this is the pretrain process')
parser.add_argument('-metric_learning',type=bool,default=False,help='if this is the pretrain process')
args = parser.parse_args()
pprint(vars(args))

#transform str parameter into list
if args.feature_pyramid is not None:
    args.feature_pyramid = [int(x) for x in args.feature_pyramid.split(',')]
args.patch_list = [int(x) for x in args.patch_list.split(',')]


num_gpu = set_gpu(args)
set_seed(args.seed)

dataset_name = args.dataset

args.save_path = 'pre_train/%s/%d-%.4f-%d-%.2f/' % \
                 (dataset_name, args.bs, args.lr, args.step_size, args.gamma)
args.save_path = osp.join('checkpoint', args.save_path)
if args.extra_dir is not None:
    args.save_path=osp.join(args.save_path,args.extra_dir)
ensure_path(args.save_path)

args.dir = 'pretrained_model/miniimagenet/max_acc.pth'

Dataset=set_up_datasets(args)
trainset = Dataset('train', args)
train_loader = DataLoader(dataset=trainset, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True)

valset = Dataset(args.set, args)
val_sampler = CategoriesSampler(valset.label, args.num_episode, args.way, args.shot + args.query)
val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
if not args.random_val_task:
    print('fix val set for all epochs')
    val_loader = [x for x in val_loader]
print('save all checkpoint models:', (args.save_all is True))

model = Proto(args, mode='pre_train')
# model = nn.DataParallel(model, list(range(num_gpu)))
model = model.cuda()

# label of query images.
label = torch.arange(args.way, dtype=torch.int8).repeat(args.query)  # shape[75]:012340123401234...
label = label.type(torch.LongTensor)
label = label.cuda()

optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}], momentum=0.9, nesterov=True, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


def save_model(name):
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))


trlog = {}
trlog['args'] = vars(args)
trlog['train_loss'] = []
trlog['val_loss'] = []
trlog['train_acc'] = []
trlog['val_acc'] = []
trlog['max_acc'] = 0.0
trlog['max_acc_epoch'] = 0

global_count = 0
writer = SummaryWriter(osp.join(args.save_path, 'tf'))

result_list = [args.save_path]
for epoch in range(1, args.max_epoch + 1):
    print(args.save_path)
    start_time = time.time()
    model = model.train()
    
    tl = Averager()
    ta = Averager()
    #standard classification for pretrain
    tqdm_gen = tqdm.tqdm(train_loader)
    for i, batch in enumerate(tqdm_gen, 1):
        global_count = global_count + 1
        if args.use_gpu:
            data, train_label = [_.cuda() for _ in batch]
        model.mode = 'pre_train'
        logits = model(data)
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        writer.add_scalar('data/loss', float(loss), global_count)
        writer.add_scalar('data/acc', float(acc), global_count)
        total_loss = loss
        writer.add_scalar('data/total_loss', float(total_loss), global_count)
        tqdm_gen.set_description('epo {}, total loss={:.4f} acc={:.4f}'.format(epoch, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()

    model = model.eval()
    
    vl = Averager()
    va = Averager()
    #use deepemd fcn for validation
    with torch.no_grad():
        tqdm_gen = tqdm.tqdm(val_loader)
        for i, batch in enumerate(tqdm_gen, 1):

            if args.use_gpu:
                data, train_label = [_.cuda() for _ in batch]

            model.mode = 'train_proto'
            logit,metric_loss = model(data)

            # loss = F.cross_entropy(logit, label) + metric_loss
            loss = F.cross_entropy(logit, label)
            acc = count_acc(logit, label)
            vl.add(loss.item())
            va.add(acc)

        vl = vl.item()
        va = va.item()
    writer.add_scalar('data/val_loss', float(vl), epoch)
    writer.add_scalar('data/val_acc', float(va), epoch)
    tqdm_gen.set_description('epo {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

    if va >= trlog['max_acc']:
        print('A better model is found!!')
        trlog['max_acc'] = va
        trlog['max_acc_epoch'] = epoch
        save_model('max_acc')
        torch.save(optimizer.state_dict(), osp.join(args.save_path, 'optimizer_best.pth'))

    trlog['train_loss'].append(tl)
    trlog['train_acc'].append(ta)
    trlog['val_loss'].append(vl)
    trlog['val_acc'].append(va)

    result_list.append(
        'epoch:%03d,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (epoch, tl, ta, vl, va))
    torch.save(trlog, osp.join(args.save_path, 'trlog'))
    if args.save_all:
        save_model('epoch-%d' % epoch)
        torch.save(optimizer.state_dict(), osp.join(args.save_path, 'optimizer_latest.pth'))
    print('best epoch {}, best val acc={:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']))
    print('This epoch takes %d seconds' % (time.time() - start_time),
          '\nstill need around %.2f hour to finish' % ((time.time() - start_time) * (args.max_epoch - epoch) / 3600))
    lr_scheduler.step()

writer.close()
result_list.append('Val Best Epoch {},\nbest val Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc'], ))
save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)
