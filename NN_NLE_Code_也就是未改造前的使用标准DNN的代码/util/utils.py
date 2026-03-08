import os
import shutil
import torch
from torch.nn import init
import torch.nn as nn


def load_checkpoint(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('[*] Loading checkpoint succeeds! Copy variables from % s! ' % ckpt_path)
    return ckpt, ckpt_path


def load_checkpoint1(ckpt_dir_or_file, map_location=None, load_best=False):
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
        else:
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint1')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('[*] Loading checkpoint succeeds! Copy variables from % s! ' % ckpt_path)
    return ckpt


# save checkpoint
def save_checkpoint(obj, save_path, is_best=False, max_keep=None):
    torch.save(obj, save_path)
    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint')

    # [修复] 不要覆盖 save_path，使用新变量 filename 存储文件名
    filename = os.path.basename(save_path)

    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [filename + '\n'] + ckpt_list
    else:
        ckpt_list = [filename + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            # 这里的 ckpt 是文件名，需要拼接路径
            ckpt_to_del = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt_to_del):
                os.remove(ckpt_to_del)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    if is_best:
        # [修复] 这里的 save_path 现在是完整的路径，shutil.copyfile 能够找到它
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def save_checkpoint1(obj, save_path, is_best=False, max_keep=None):
    torch.save(obj, save_path)
    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint1')

    filename = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [filename + '\n'] + ckpt_list
    else:
        ckpt_list = [filename + '\n']

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt_to_del = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt_to_del):
                os.remove(ckpt_to_del)
        ckpt_list[max_keep:] = []

    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose1d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


# [修复] 下面的 init 函数全部加上下划线 (如 normal_ 替代 normal)，解决 UserWarning
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def mkdir(paths):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)