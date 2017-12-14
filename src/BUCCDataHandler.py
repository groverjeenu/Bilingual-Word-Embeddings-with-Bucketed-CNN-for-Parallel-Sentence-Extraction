import codecs
import pickle
import random

import numpy as np


def random_bin(p):
    l = np.random.randint(0, 1000)

    if l <= int(1000 * p):
        return 1
    return 0


def samplePositiveNegative(gold, Gold_dict, F1, F2, file1, file2):  ###### File1 must be the first column in gold
    p = 0.05

    Data = []  ############ GET ALL POSITIVES
    for g in gold:
        d = []
        d.append(F1[g[0]])
        d.append(F2[g[1]])
        d.append(1)

        Data.append(d)

    ######### Naive Negative Sampling
    list_F2 = F2.keys()
    len_F2 = len(list_F2)
    for ind, j in enumerate(F1.keys()):
        print len(Data)
        if random_bin(p * (len(gold)) / len(Gold_dict.keys())):
            d = []
            val = list_F2[np.random.randint(0, len_F2)]
            try:
                if Gold_dict[j] != val:
                    d.append(F1[j])
                    d.append(F2[val])
                    d.append(0)
                    Data.append(d)
            except:
                d.append(F1[j])
                d.append(F2[val])
                d.append(0)
                Data.append(d)

    random.shuffle(Data)
    return Data


def getTrainTestValidationSampledData(file1, file2, fileg):
    train_split = 0.6
    valid_split = 0.1
    test_split = 0.3

    f = open(fileg, 'r')
    Gold = []

    Gold_dict = {}
    for line in f:
        line = line.split()
        Gold_dict[line[0]] = line[1]
        Gold.append(line)

    random.shuffle(Gold)
    l = len(Gold)

    f = codecs.open(file1, 'r', encoding='utf-8')

    F1 = {}
    for line in f:
        ind = line.find('\t')
        sent = line[ind + 1:].strip('\n')
        id_ = line[:ind]
        F1[id_] = sent

    f = codecs.open(file2, 'r', encoding='utf-8')

    F2 = {}
    for line in f:
        ind = line.find('\t')
        sent = line[ind + 1:].strip('\n')
        id_ = line[:ind]
        F2[id_] = sent

    print len(F1.keys())
    print len(F2.keys())

    train_gold = Gold[:int(train_split * l)]
    valid_gold = Gold[int(train_split * l): int((train_split + valid_split) * l)]
    test_gold = Gold[int((train_split + valid_split) * l):]

    train_data = samplePositiveNegative(train_gold, Gold_dict, F1, F2, file1, file2)
    valid_data = samplePositiveNegative(valid_gold, Gold_dict, F1, F2, file1, file2)
    test_data = samplePositiveNegative(test_gold, Gold_dict, F1, F2, file1, file2)

    for i in range(0, 10):
        print train_data[i]

    dump = []
    dump.append(train_data)
    dump.append(valid_data)
    dump.append(test_data)

    filed = open('../data/data/bucc2017/de-en/train_valid_test/data', 'wb')
    pickle.dump(dump, filed)
    filed.close()

    return train_data, valid_data, test_data


def loadTrainTestValidationSampledData(file1):
    filed = open(file1, 'rb')
    data = pickle.load(filed)
    filed.close()
    train_data, valid_data, test_data = data[0], data[1], data[2]
    return train_data, valid_data, test_data


def main():
    # Uncomment the line below to sample the data again. In github, just uploading the sampled data due to size constraints,
    # If you want to resample the data again, download whole dataset from the BUCC2017 website and place in te directory structure as mentioned in the paths below.

    # getTrainTestValidationSampledData('../data/data/bucc2017/de-en/de-en.training.de','../data/data/bucc2017/de-en/de-en.training.en','../data/data/bucc2017/de-en/de-en.training.gold')
    x, y, z = loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')


if __name__ == "__main__":
    main()
