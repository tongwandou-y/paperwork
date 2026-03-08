from util.load_data import *
from sklearn.metrics import mean_squared_error
import configs as cfg
import shutil
def n_mse(data1,data2):
    NM = len(data1)
    data3 = mean_squared_error(data1, data2)
    data3 = data3 * NM
    data1_1 = map(lambda x: x ** 2, data1)
    data1_sum = sum(data1_1)
    data4 = data3 / data1_sum
    return data4

def mse_score(filename1,filename2):
    N1 = load_data_cvs5(filename2)
    # N1 = len(aaa)
    data1 = load_data_cvs3(filename1, 1, N1+400,0)
    data1 = data1[400:]
    data2 = load_data_cvs3(filename2, 1, N1,0)
    # data2 = data2[:-10000]
    data3 = mean_squared_error(data1, data2)
    data4 = n_mse(data1, data2)
    print(data3)
    print(data4)
    return data4


if __name__ == '__main__':
    cfgs = cfg.Configs['qam8_4g_test']
    mse_score = mse_score(filename1=cfgs.filename2, filename2='./out1.csv')
    mse_score = round(mse_score[0], 7)
    mse_score_str = str(mse_score)

    source1 = './result.png'
    target1 = './'+mse_score_str+'.png'
    shutil.copy(source1, target1)

    source2 = './1/out1.csv'
    target2 = './'+mse_score_str+'.csv'
    shutil.copy(source2, target2)

