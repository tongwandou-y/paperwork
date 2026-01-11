from torch.utils.data import DataLoader
from util.load_data import *
from util.utils import *
from models.model.transformer import Transformer
from models.model.DNN import DNN
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import configs as cfg
import time
import os
import shutil
plt.rcParams['axes.linewidth'] = 1.5  # 图框宽度

font = {'family': 'DejaVu Sans',
        # 'weight': 'normal',
        'weight': 'bold',
        'size': 12}
font1 = {'family': 'DejaVu Sans',
        # 'weight': 'normal',
        'weight': 'bold',
        'size': 18}
plt.rc('font', **font)


def n_mse(data1,data2):
    NM = len(data1)
    data3 = mean_squared_error(data1, data2)
    data3 = data3 * NM
    data1_1 = map(lambda x: x ** 2, data1)
    data1_sum = sum(data1_1)
    data4 = data3 / data1_sum
    return data4

def mse_score(filename1,filename2,cfgs):
    N1 = load_data_cvs5(filename2)
    # N1 = len(aaa)
    d_model = cfgs.d_model
    seq_len = cfgs.seq_len
    aaaa = d_model*seq_len
    data1 = load_data_cvs3(filename1, 1, N1+aaaa,0)
    data1 = data1[aaaa:]
    data2 = load_data_cvs3(filename2, 1, N1,0)
    # data2 = data2[:-10000]
    data3 = mean_squared_error(data1, data2)
    data4 = n_mse(data1, data2)
    print(data3)
    print(data4)
    return data4

def copyfile(source):
    target1 = './test_weight'
    target = os.path.join(target1, os.path.dirname(source))
    if not os.path.isdir(target):
        os.makedirs(target)
    shutil.copy(source, target)


def test(configs):
    device = configs.device
    #model = DNN(configs).to(device)
    model = Transformer(configs).to(device)
    model.eval()
    init_weights(model, init_type=configs.init_type)
    start_time = time.time()
    ckpt_dir = './output/%s/checkpoint' % configs.experiment_name
    ckpt, ckpt_path = load_checkpoint(ckpt_dir)
    copyfile(ckpt_path)
    model.load_state_dict(ckpt['model'])
    samples = CVSDataset(configs.filename1,
                         configs.filename2,
                         configs.seq_len,
                         configs.d_model)
    trainloader = DataLoader(
        dataset=samples,
        batch_size=configs.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )
    tgt = torch.Tensor([])
    for i, (before_data_a, before_data_c,after_data_) in enumerate(trainloader):
        src = before_data_a.to(device)
        trg = before_data_c.to(device)
        #src = src.reshape(configs.batch_size, -1)
        tgt_ = model(src, trg).detach().cpu()
        tgt_c = tgt_.contiguous().view(-1)
        tgt = torch.cat((tgt, tgt_c), 0)

    end_time = time.time()
    print('All %f second' % (end_time - start_time))
    tgt = tgt.cpu().numpy()
    N1 = len(tgt)
    time_data = load_data_cvs4(configs.filename1, 0, N1, 0)
    time_power_data = data_connect(time_data, tgt)
    data_writer(configs.filename3, time_power_data)
    return ckpt_path

def data_draw7(filename1,filename2,N1,cfgs):
    # 英文；三图合并到一起的子程序
    d_model = cfgs.d_model
    seq_len = cfgs.seq_len
    aaaa = d_model*seq_len
    time = load_data_cvs4(filename1, 0, N1,0)
    data1 = load_data_cvs3(filename1, 1, N1+aaaa,0)
    data1 = data1[aaaa:]
    data2 = load_data_cvs3(filename2, 1, N1,0)
    plt.figure(figsize=(16, 9),dpi=100)
    plt.subplot(3,1,1)
    plt.plot(time,data1,color="steelblue",linewidth=1.5,label = "Real waveform")
    plt.legend(loc='upper right')
    plt.ylabel("Amplitude(mV)",font1)
    plt.xlim(2.5e-6, 2.55e-6)
    plt.ylim(0, 1)
    plt.subplot(3,1,2)
    plt.axis([2.5e-6, 2.55e-6, 0, 1])
    plt.plot(time, data2, color="brown", linewidth=1.5, label="Generated fitting waveform")
    plt.legend(loc='upper right')
    plt.ylabel("Amplitude(mV)",font1)
    plt.xlim(2.5e-6, 2.55e-6)
    plt.ylim(0, 1)
    plt.subplot(3,1,3)
    plt.axis([2.51e-6, 2.52e-6, 0, 1])
    plt.plot(time, data1, color="steelblue", linewidth=3, linestyle="--", label="Real waveform")
    plt.plot(time, data2, color="brown", linewidth=1.5, label="Generated fitting waveform")
    plt.ylabel("Amplitude(mV)",font1)
    plt.xlabel("Time(s)",font1)
    plt.legend(loc='upper right')
    plt.xlim(2.51e-6, 2.52e-6)
    plt.ylim(0, 1)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.98,
                    wspace=None, hspace=None) # 更改子图之间的距离
    plt.savefig('result.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    cfgs = cfg.Configs['qam8_4g_test']
    ckpt_path = test(cfgs)
    mse_score = mse_score(filename1=cfgs.filename2, filename2='./out1.csv', cfgs=cfgs)
    mes_save(cfgs.filename0,ckpt_path, mse_score)
    # 画图
    #filename1 = cfgs.filename2
    #filename2 = 'out1.csv'
    #aaa = load_data_cvs_plot(filename2, 1)
    #N1 = len(aaa)
    # 当你运行自己的文件时，需要调整函数data_draw7中坐标轴的范围，如果调整不好可以把这个函数注销掉。
    #data_draw7(filename1, filename2, N1, cfgs)