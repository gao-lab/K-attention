import glob
import pdb
import random
import numpy as np
import pandas as pd
import h5py
import os
from sklearn.model_selection import StratifiedKFold


def read_csv(path):
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0]!="Type"]

    Type  = 0
    loc   = 1
    Seq   = 2
    Str   = 3
    Score = 4
    label = 5

    rnac_set  = df[Type].to_numpy()
    sequences = df[Seq].to_numpy()
    structs  = df[Str].to_numpy()
    targets   = df[Score].to_numpy().astype(np.float32).reshape(-1,1)
    return sequences, structs, targets

def seq_to_matrix(seq, seq_matrix, seq_order):
    '''
    change target 3D tensor according to sequence and order
    :param seq: 输入的单根序列
    :param seq_matrix: 输入的初始化的矩阵
    :param seq_order:这是第几个序列
    :return:
    '''
    for i in range(len(seq)):
        if ((seq[i] == 'A') | (seq[i] == 'a')):
            seq_matrix[seq_order, i, 0] = 1
        if ((seq[i] == 'C') | (seq[i] == 'c')):
            seq_matrix[seq_order, i, 1] = 1
        if ((seq[i] == 'G') | (seq[i] == 'g')):
            seq_matrix[seq_order, i, 2] = 1
        if ((seq[i] == 'T') | (seq[i] == 't')):
            seq_matrix[seq_order, i, 3] = 1
    return seq_matrix


def strtomatrix(structs):
    """

    Args:
        structs:

    Returns:

    """
    structure = np.zeros((len(structs), 1, 101))
    for i in range(len(structs)):
        struct = structs[i].split(',')
        ti = [float(t) for t in struct]
        ti = np.array(ti).reshape(1, -1)
        structure[i] = np.concatenate([ti], axis=0)
    return structure

def cross_validation(number_of_folds, total_number, random_seeds=233):
    """
    :param number_of_folds:
    :param total_number:
    :param random_seeds:
    :return:
    """
    x = np.zeros((total_number,), dtype=np.int)
    split_iterator = StratifiedKFold(n_splits=number_of_folds, random_state=random_seeds, shuffle=True)
    split_train_index_and_test_index_list = [
        (train_index, test_index)
        for train_index, test_index in split_iterator.split(x, x)
    ]
    return (split_train_index_and_test_index_list)


def split_dataset(split_index_list, fold, data_x, data_y, data_id=None):
    """
    generate training dataset and test data set
    :param split_index_list:
    :param fold:
    :param data_id:
    :param data_x:X
    :param data_y:Y
    :return:
    """
    id_train = data_id[split_index_list[fold][0].tolist()]
    x_train = data_x[split_index_list[fold][0].tolist()]
    y_train = data_y[split_index_list[fold][0].tolist()]
    id_test = data_id[split_index_list[fold][1].tolist()]
    x_test = data_x[split_index_list[fold][1].tolist()]
    y_test = data_y[split_index_list[fold][1].tolist()]
    return [x_train, y_train, id_train, x_test, y_test, id_test]


def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)

def StoreTrainSet(rootPath, allData, ValNum=10, RandomSeeds=233):
    """
    store different dataset
    :param rootPath:
    :param ValNum: all data size /test size
    :param RandomSeeds: for generating testing dataset
    :param allData: All data
    """
    dataNum = allData[1].shape[0]
    split_train_index_and_test_index_list = cross_validation(number_of_folds=ValNum, total_number=dataNum,
                                                             random_seeds=RandomSeeds)
    i = 0
    outDataTem = split_dataset(split_train_index_and_test_index_list, fold=i, data_x=allData[0], data_y=allData[1],
                               data_id=allData[2])

    mkdir(rootPath)
    training_path = rootPath + "/train.hdf5"
    test_path = rootPath + "/test.hdf5"

    f_train = h5py.File(training_path)
    f_test = h5py.File(test_path)

    f_train.create_dataset("sequences", data=outDataTem[0])
    f_train.create_dataset("labs", data=outDataTem[1])
    f_train.create_dataset("seq_struct", data=outDataTem[2])
    f_train.close()
    print("Train: ", outDataTem[0].shape)

    f_test.create_dataset("sequences", data=outDataTem[3])
    f_test.create_dataset("labs", data=outDataTem[4])
    f_test.create_dataset("seq_struct", data=outDataTem[5])
    f_test.close()
    print("test: ", outDataTem[3].shape)


def main():
    """

    Returns:

    """
    datapath = "../clip_data/"
    filelist = glob.glob(datapath+"/*.tsv")
    for path in filelist:
        name = path.split("/")[-1].split(".")[0]
        sequences, structs, targets = read_csv(path)
        sequencesMatrix = np.zeros((len(sequences),101,4))
        targets[targets < 0] = 0
        targets[targets > 0] = 1
        for seq_order in range(len(sequences)):
            sequencesMatrix = seq_to_matrix(sequences[seq_order], sequencesMatrix, seq_order)

        structure= strtomatrix(structs)
        index_shuffle = list(range(len(sequences)))
        random.shuffle(index_shuffle)

        rootPath = "../HDF5/"+name+"/"
        mkdir(rootPath)
        allData = [sequencesMatrix, targets, structure]
        print(rootPath)
        StoreTrainSet(rootPath, allData, ValNum=5, RandomSeeds=233)








if __name__ == '__main__':
    main()


