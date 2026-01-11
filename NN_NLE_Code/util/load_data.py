# coding=utf-8   #默认编码格式为utf-8
import dask.dataframe as dd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler()

def normalization_2(data,N, m):
    aaa1 = data[0:N,m]
    aaa1 = np.array(aaa1)
    return aaa1

def normalization_3(data,N, m):
    aaa1 = data[0:N,m]
    aaa1 = np.array(aaa1)
    aaa1 = mm.fit_transform(aaa1.reshape(-1,1))
    return aaa1

def denomalization(data):
    aaa2 = mm.inverse_transform(data)
    return aaa2



def load_data_cvs(filename, seq_len, d_model):
    path = './1/'
    filenames = os.path.join(path, filename)
    data = dd.read_csv(filenames, header=0, encoding= 'gbk')
    aaa1 = np.array(data)
    N_ = len(aaa1)
    N = d_model * (N_ //  d_model)
    bbb1_ = normalization_3(aaa1, N, 1)
    bbb1 = bbb1_.reshape([-1, d_model])
    bbb1 = bbb1[seq_len:-seq_len,:]
    aaa = torch.from_numpy(bbb1).float()
    aaa = aaa.reshape([-1, 1, d_model])
    return aaa

def load_data_cvs_sliding_a(filename, seq_len, d_model):
    # [batch_size, seq_len*2 + 1, d_model]
    path = './1/'
    filenames = os.path.join(path, filename)
    data = dd.read_csv(filenames, header=0, encoding= 'gbk')
    aaa1 = np.array(data)
    N_ = len(aaa1)
    N = d_model * (N_ //  d_model)
    bbb1_ = normalization_3(aaa1, N, 1)
    bbb1,ccc1 = sliding_connection(bbb1_, seq_len, d_model)
    aaa = torch.from_numpy(bbb1).float()
    print(aaa.shape)
    return aaa

def load_data_cvs_sliding_c(filename, seq_len, d_model):
    # [batch_size, 1, d_model]
    path = './1/'
    filenames = os.path.join(path, filename)
    data = dd.read_csv(filenames, header=0, encoding= 'gbk')
    aaa1 = np.array(data)
    N_ = len(aaa1)
    N = d_model * (N_ //  d_model)
    bbb1_ = normalization_3(aaa1, N, 1)
    bbb1,ccc1 = sliding_connection(bbb1_, seq_len, d_model)
    ccc = ccc1.reshape([-1, 1, d_model])
    ccc = torch.from_numpy(ccc).float()
    return ccc

def sliding_connection(aaa, seq_len, d_model):
    aaa = aaa.reshape([-1, d_model])
    NN1 = len(aaa)-2*seq_len
    bbb = np.zeros((NN1,2*seq_len+1, d_model))
    ccc = np.zeros((NN1, d_model))
    for i in range(NN1):
        bbb_1 = np.zeros((seq_len,d_model))
        bbb_2 = np.zeros((seq_len,d_model))
        for j in range(seq_len):
            bbb_1[seq_len-j-1,:] = aaa[i+seq_len-1-j]
            bbb_2[j,:] = aaa[i+seq_len+1+j]
        kk = aaa[i+seq_len]
        ccc[i] = kk
        kk = kk.reshape(1,d_model)
        bbb[i,:,:] = np.concatenate((bbb_1, kk, bbb_2), axis=0)
    return bbb, ccc



def load_data_cvs3(filename, m, N1, M1):
    path = './1/'
    filenames = os.path.join(path, filename)
    data = dd.read_csv(filenames,header=M1, encoding= 'gbk')
    aaa1 = np.array(data)
    aaa = normalization_3(aaa1,N1, m)
    aaa = np.array(aaa)
    return aaa

def load_data_cvs4(filename, m, N1,M1):
    path = './1/'
    filenames = os.path.join(path, filename)
    data = dd.read_csv(filenames,header=M1, encoding= 'gbk')
    aaa1 = np.array(data)
    aaa = normalization_2(aaa1,N1, m)
    aaa = np.array(aaa)
    return aaa

def load_data_cvs5(filename):
    path = './1/'
    filenames = os.path.join(path, filename)
    data = dd.read_csv(filenames, header=0, encoding= 'gbk')
    aaa1 = np.array(data)
    N_ = len(aaa1)
    return N_

class CVSDataset(Dataset):

    def __init__(self, before_file, after_file, seq_len, d_model):
        self.load_before_data_a = load_data_cvs_sliding_a(before_file, seq_len, d_model)
        self.load_before_data_c = load_data_cvs_sliding_c(before_file, seq_len, d_model)
        self.load_after_data = load_data_cvs(after_file, seq_len, d_model)

    def __len__(self):
        return len(self.load_after_data)

    def __getitem__(self, index):
        load_before_data_a = self.load_before_data_a[index]
        load_before_data_c = self.load_before_data_c[index]
        load_after_data = self.load_after_data[index]

        return load_before_data_a,load_before_data_c, load_after_data


def load_data_cvs_plot(filename, NN):
    path = './1/'
    filenames = os.path.join(path, filename)
    print("os.path.join")
    data = dd.read_csv(filenames, header=5, encoding= 'gbk')
    print("dd.read_csv")
    aaa1 = np.array(data)
    N_ = len(aaa1)
    print(N_)
    N = NN * (N_ // NN)
    aaa = normalization_3(aaa1,N, 1)
    aaa = np.array(aaa)
    AA = aaa.reshape([-1,1,NN])
    aaa = torch.from_numpy(AA).float()
    return aaa

def data_writer(file, time_power_data):
    with open(file, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["time", "power"])
        rowcount = np.shape(time_power_data)[0]
        for i in range(rowcount):
            writer.writerow(time_power_data[i])
        csvfile.close()



def data_connect(time_data, power_data):
    time_data_ = np.array(time_data)
    power_data_ = np.array(power_data)
    time_data_1 = time_data_.reshape(time_data_.shape[0], 1)
    power_data_1 = power_data_.reshape(power_data_.shape[0], 1)
    time_power_data = np.hstack((time_data_1, power_data_1))
    return time_power_data

def mes_save(filename, ckpt_path,mse_score):
    file = open(filename, 'a')
    file.write('%s        %.8f '%(ckpt_path[33:],mse_score))
    file.write('\n')
    file.close()



# if __name__ == '__main__':
    # main()