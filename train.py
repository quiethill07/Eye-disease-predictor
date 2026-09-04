import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

import archs

import losses
from dataset import Dataset

from metrics import iou_score, indicators

from utils import AverageMeter, str2bool

try:
    from tensorboardX import SummaryWriter
except ImportError:
    # Fallback for environments where tensorboardX is unavailable.
    from torch.utils.tensorboard import SummaryWriter

import shutil
import os
import subprocess

from pdb import set_trace as st


ARCH_NAMES = getattr(archs, '__all__', [k for k, v in archs.__dict__.items() if callable(v)])
LOSS_NAMES = list(getattr(losses, '__all__', [k for k, v in losses.__dict__.items() if callable(v)]))
if 'BCEWithLogitsLoss' not in LOSS_NAMES:
    LOSS_NAMES.append('BCEWithLogitsLoss')


def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2981, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UKAN')
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='fives', help='dataset name')      
    parser.add_argument('--data_dir', default='inputs', help='dataset dir')
    parser.add_argument('--val_ratio', default=0.2, type=float, help='validation ratio from train split')

    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--resume_checkpoint', default='', type=str, help='path to full training checkpoint to resume from')
    parser.add_argument('--save_freq', default=5, type=int, help='save a full checkpoint every N epochs')
    parser.add_argument('--save_last', default=True, type=str2bool, help='save last checkpoint every epoch')
    parser.add_argument('--save_best_full', default=True, type=str2bool, help='save full best checkpoint for training resume')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=20, type=int,
                        metavar='N', help='early stopping patience in epochs (default: 20)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--disable_tqdm', default=False, type=str2bool, help='disable per-batch tqdm bars')
    parser.add_argument('--epoch_progress', default=True, type=str2bool, help='show a single epoch-level progress bar')

    parser.add_argument('--no_kan', action='store_true')
    parser.add_argument('--attention_mode', default='none', choices=['none', 'se_only', 'cbam_only', 'cbam_se'],
                        help='attention module setting for UKAN')



    config = parser.parse_args()

    return config


def train(config, train_loader, model, criterion, optimizer, device, epoch=None, total_epochs=None):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    data_iter = train_loader
    pbar = None
    if not config.get('disable_tqdm', False):
        if epoch is not None and total_epochs is not None:
            desc = f'Train {epoch + 1}/{total_epochs}'
        else:
            desc = 'Train'
        pbar = tqdm(total=len(train_loader), desc=desc, leave=False, mininterval=0.5)

    for input, target, _ in data_iter:
        non_blocking = device.type == 'cuda'
        input = input.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # compute output
        if config['deep_supervision']:
            outputs = model(input)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)

            iou, dice, _ = iou_score(outputs[-1], target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(outputs[-1], target)
            
        else:
            output = model(input)
            loss = criterion(output, target)
            iou, dice, _ = iou_score(output, target)
            iou_, dice_, hd_, hd95_, recall_, specificity_, precision_ = indicators(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice', avg_meters['dice'].avg),
        ])
        if pbar is not None:
            pbar.set_postfix(postfix)
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])


def validate(config, val_loader, model, criterion, device, epoch=None, total_epochs=None):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        data_iter = val_loader
        pbar = None
        if not config.get('disable_tqdm', False):
            if epoch is not None and total_epochs is not None:
                desc = f'Val {epoch + 1}/{total_epochs}'
            else:
                desc = 'Val'
            pbar = tqdm(total=len(val_loader), desc=desc, leave=False, mininterval=0.5)
        for input, target, _ in data_iter:
            non_blocking = device.type == 'cuda'
            input = input.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                if not isinstance(outputs, (list, tuple)):
                    outputs = [outputs]
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou, dice, _ = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou, dice, _ = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            if pbar is not None:
                pbar.set_postfix(postfix)
                pbar.update(1)
        if pbar is not None:
            pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def torch_load_compat(path, map_location='cpu'):
    """Compatibility loader for PyTorch>=2.6 (weights_only default changed)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older torch versions don't support weights_only argument.
        return torch.load(path, map_location=map_location)


def load_resume_checkpoint(model, optimizer, scheduler, ckpt_path):
    if not ckpt_path:
        return 0, 0.0, 0.0

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Resume checkpoint not found: {ckpt_path}')

    ckpt = torch_load_compat(ckpt_path, map_location='cpu')
    if not isinstance(ckpt, dict):
        raise RuntimeError('Unsupported resume checkpoint format.')
    if 'model' not in ckpt:
        raise RuntimeError('Resume checkpoint missing model state.')

    incompatible = model.load_state_dict(ckpt['model'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f'WARNING: resume checkpoint does not fully match the model.\n'
            f'  Missing (randomly initialized): {list(incompatible.missing_keys)}\n'
            f'  Unexpected (ignored): {list(incompatible.unexpected_keys)}'
        )
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler is not None and ckpt.get('scheduler') is not None:
        scheduler.load_state_dict(ckpt['scheduler'])

    start_epoch = int(ckpt.get('epoch', -1)) + 1
    best_iou = float(ckpt.get('best_iou', 0.0))
    best_dice = float(ckpt.get('best_dice', 0.0))
    print(f'=> resumed from checkpoint: {ckpt_path}')
    print(f'=> start_epoch: {start_epoch}, best_iou: {best_iou:.4f}, best_dice: {best_dice:.4f}')
    return start_epoch, best_iou, best_dice


def make_checkpoint_payload(epoch, model, optimizer, scheduler, best_iou, best_dice, config):
    payload = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_iou': best_iou,
        'best_dice': best_dice,
        'config': config,
    }
    return payload


def main():
    seed_torch()
    config = vars(parse_args())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    exp_name = config['name']
    output_dir = config.get('output_dir')

    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    cudnn.benchmark = device.type == 'cuda'

    # create model
    model = archs.__dict__[config['arch']](
        config['num_classes'],
        config['input_channels'],
        config['deep_supervision'],
        embed_dims=config['input_list'],
        no_kan=config['no_kan'],
        attention_mode=config.get('attention_mode', 'none'),
    )

    model = model.to(device)
    param_groups = []

    kan_fc_params = []
    other_params = []

    for name, param in model.named_parameters():
        # print(name, "=>", param.shape)
        if 'layer' in name.lower() and 'fc' in name.lower(): # higher lr for kan layers
            # kan_fc_params.append(name)
            param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay'], 'is_kan': True}) 
        else:
            # other_params.append(name)
            param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay'], 'is_kan': False})  
    

    
    # st()
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(param_groups)


    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Resolve relative to this file so the run can be launched from any cwd.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(os.path.join(script_dir, 'train.py'), f'{output_dir}/{exp_name}/')
    shutil.copy2(os.path.join(script_dir, 'archs.py'), f'{output_dir}/{exp_name}/')

    dataset_name = config['dataset'].lower()
    img_ext = '.png'
    mask_ext = '.png'
    mask_in_class_subdir = True

    # Data loading code
    if dataset_name == 'fives':
        train_original_dir = os.path.join(config['data_dir'], 'train', 'Original')
        train_gt_dir = os.path.join(config['data_dir'], 'train', 'Ground truth')
        mask_in_class_subdir = False

        img_ids = sorted(glob(os.path.join(train_original_dir, '*' + img_ext)))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=config['val_ratio'], random_state=config['dataseed'])

        train_img_dir = train_original_dir
        train_mask_dir = train_gt_dir
        val_img_dir = train_original_dir
        val_mask_dir = train_gt_dir
    else:
        img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'], 'images', '*' + img_ext)))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
        train_img_ids, val_img_ids = train_test_split(img_ids, test_size=config['val_ratio'], random_state=config['dataseed'])

        train_img_dir = os.path.join(config['data_dir'], config['dataset'], 'images')
        train_mask_dir = os.path.join(config['data_dir'], config['dataset'], 'masks')
        val_img_dir = train_img_dir
        val_mask_dir = train_mask_dir

    train_transform = A.Compose([
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    val_transform = A.Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])

    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=train_transform,
        mask_in_class_subdir=mask_in_class_subdir)
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        img_ext=img_ext,
        mask_ext=mask_ext,
        num_classes=config['num_classes'],
        transform=val_transform,
        mask_in_class_subdir=mask_in_class_subdir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])


    start_epoch = 0
    best_iou = 0
    best_dice= 0
    if config['resume_checkpoint']:
        start_epoch, best_iou, best_dice = load_resume_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=config['resume_checkpoint'],
        )

    trigger = 0
    for epoch in range(start_epoch, config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer, device=device, epoch=epoch, total_epochs=config['epochs'])
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion, device=device, epoch=epoch, total_epochs=config['epochs'])

        # MultiStepLR was previously constructed but never stepped, so its
        # milestones never fired and the LR stayed constant for the whole run.
        if config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        elif scheduler is not None:
            scheduler.step()

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice'], val_log['loss'], val_log['iou'], val_log['dice']))

        log['epoch'].append(epoch)
        # Log the LR actually in effect, not the static config value; with a
        # scheduler the two diverge immediately.
        log['lr'].append(optimizer.param_groups[0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

        my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
        my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
        my_writer.add_scalar('train/dice', train_log['dice'], global_step=epoch)
        my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
        my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
        my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

        my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
        my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model.pth')
            best_iou = val_log['iou']
            best_dice = val_log['dice']
            print("=> saved best model")
            print('train_loss: %.4f | val_loss: %.4f' % (train_log['loss'], val_log['loss']))
            print('IoU: %.4f' % best_iou)
            print('Dice: %.4f' % best_dice)
            if config['save_best_full']:
                best_ckpt = make_checkpoint_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_iou=best_iou,
                    best_dice=best_dice,
                    config=config,
                )
                torch.save(best_ckpt, f'{output_dir}/{exp_name}/best_checkpoint.pth')
            trigger = 0

        # periodic full checkpoints for resume/debugging
        if config['save_freq'] > 0 and (epoch + 1) % config['save_freq'] == 0:
            periodic_ckpt = make_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_iou=best_iou,
                best_dice=best_dice,
                config=config,
            )
            ckpt_path = f'{output_dir}/{exp_name}/checkpoint_epoch_{epoch + 1}.pth'
            torch.save(periodic_ckpt, ckpt_path)
            print(f"=> saved periodic checkpoint: {ckpt_path}")

        # always keep a resumable "last" checkpoint
        if config['save_last']:
            last_ckpt = make_checkpoint_payload(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_iou=best_iou,
                best_dice=best_dice,
                config=config,
            )
            torch.save(last_ckpt, f'{output_dir}/{exp_name}/last_checkpoint.pth')

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()
