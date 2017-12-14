import codecs

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def loadWordvecs(file):
    f = f = codecs.open(file, 'r', encoding='utf-8')
    cnt = 0
    En = {}
    for line in f:
        if (cnt > 0):
            line = line.split()
            En[line[0]] = np.array([np.float(j) for j in line[1:]]).reshape(1, -1)
        cnt = cnt + 1

    return En


def closestVector(L1, L2, word):
    ##Assumes word belongs to language L1
    vec = 0

    try:
        vec = L1[word]
    except:
        print word + " not found in L1"

    Similar = {}

    maxe = -2
    match = ''
    for key in L2.keys():
        Similar[key] = cosine_similarity(vec, L2[key])[0, 0]
        if Similar[key] > maxe:
            maxe = Similar[key]
            match = key
    print maxe, match

    for ind, w in enumerate(sorted(Similar, key=Similar.get, reverse=True)):
        print w, Similar[w]
        if (ind == 9):
            break

    print L1[word]
    print L2[match]
    print cosine_similarity(L1[word], L2[match])


def main():
    En = loadWordvecs('../data/unsup.128.en')
    De = loadWordvecs('../data/unsup.128.de')
    closestVector(En, De, 'book')


if __name__ == "__main__":
    main()
