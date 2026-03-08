import dask.dataframe as dd
import os
import numpy as np
import torch
import time
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from util.load_data import *
from sklearn.metrics import mean_squared_error

def data_draw(filename):

    time1 = load_data_cvs(filename, 0)
    data = load_data_cvs(filename, 1)
    plt.figure(1)
    plt.plot(time1, data)
    # plt.xlim(0,3e-5)
    plt.show()

def n_mse(data1, data2):
    NM = len(data1)
    data3 = mean_squared_error(data1,data2)
    data3 = data3 * NM
    data1_1 = map(lambda x:x**2, data1)
    data1_sum = sum(data1_1)
    data4 = data3/data1_sum
    return data4


def data_draw2(filename1, filename2,N1):

    time = load_data_cvs4(filename1, 0, N1,0)
    data1 = load_data_cvs3(filename1, 1, N1+600,0)
    data1 = data1[600:]
    # data1 = load_data_cvs3(filename1, 1, N1,0)
    # data1 = data1[600:]
    data2 = load_data_cvs3(filename2, 1, N1,0)
    data3 = mean_squared_error(data1, data2)
    data4 = n_mse(data1, data2)
    print(data3)
    print(data4)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(time, data1, 'g-',linestyle="--",label="真实曲线")
    ax2.plot(time, data2, 'r-',label="拟合曲线")
    plt.xlim(0, 8e-9)
    # plt.xlim(690e-5,695e-5)
    plt.show()

if __name__ == '__main__':
    # aaa = load_data_cvs(filename,N)
    # torch.set_printoptions(precision=16)
    # print(aaa[0:10])
    # data_draw(filename)
    # filename = 'after_fiber_data_b.csv'
    filename1 = 'Rx-pd-qam16_25km_awg64g_power10_symr2g_Rf8g_num1e4_SNRno_Randi_1-evm9.669222-tb1-3.csv'
    # filename1 = 'Rx-pd-qam16_25km_awg64g_power10_symr2g_Rf8g_num1e4_SNRno_PRBS23_1-evm9.400126-tb1-3.csv'
    # filename1 = 'Tx-qam16_25KM_symr2g_Rf8g_num1e4_SNRno_Randi_1-1.csv'
    filename2 = 'out1.csv'
    # filename2 = 'Rx-pd-qam16_25km_awg64g_power10_symr2g_Rf8g_num1e4_SNRno_Randi_1-evm9.669222-tb1-3.csv'
    aaa = load_data_cvs_plot(filename2, 1)
    N1 = len(aaa)
    data_draw2(filename1, filename2, N1)



