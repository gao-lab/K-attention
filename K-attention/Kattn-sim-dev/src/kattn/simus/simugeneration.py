import numpy as np
import pdb
import os
import h5py
import random
from sklearn.model_selection import StratifiedKFold
import math
import time
from matplotlib import pyplot as plt
# import seaborn as sns
import pandas as pd
plt.switch_backend('agg')
import glob
import pdb
import argparse
import sys
from multiprocessing import Pool

def run_model(position, feature):

    cmd = "python -u simu_main.py"

    tmp_cmd = str(cmd + " --position " + str(position) + " --feature " + str(feature))

    os.system(tmp_cmd)

def generateParamPair():

    def get_best_parameter(dataname):
        """

        Args:
            dataname:

        Returns:

        """
    paramlist = []
    pos_list = ['absolute','random','abs-ran','relative']
    fea_list = ['pwm','rand','fix1','fix2']

    for pos in pos_list:
        for fea in fea_list:
            paramlist.append([pos,fea])
    return paramlist

if __name__ == '__main__':
    # grid search

    paramlist = generateParamPair()
    cpus = 4

    num = int(len(paramlist)/cpus)+1


    for i in range(num):

        pool = Pool(processes=cpus)

        for param in paramlist[i*cpus:min((i+1)*cpus,len(paramlist))]:
            position, feature = param
            # run_model(KernelLen, KernelNum, RandomSeed, dataname, Modeltype)
            pool.apply_async(run_model, (str(position), str(feature)))
            time.sleep(1)
        pool.close()
        pool.join()