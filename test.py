#coding:utf-8
import core.model as model
import core.train as train
import core.utils as utils
import scipy.io as sio

if __name__=='__main__':
    data = utils.load_data('./data/Train/A01T.mat',label=False)
    print(__name__)
   