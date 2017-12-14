import math

import numpy as np
import sklearn.metrics as sk


def getSimilarityMatrix(vec1, vec2, metric='cos'):
    mat = np.zeros((len(vec1), len(vec2)))

    for i in range(0, len(vec1)):
        for j in range(0, len(vec2)):
            if (metric == 'cos'):
                mat[i][j] = sk.pairwise.cosine_similarity(vec1[i], vec2[j])
            elif metric == 'euclid':
                mat[i][j] = sk.pairwise.euclidean_distances(vec1[i], vec2[j])

    return mat


def getDynamicPooledMatrix(mat, dim=15, type='max'):
    l1 = np.shape(mat)[0]
    l2 = np.shape(mat)[1]

    wide_x = wide_y = 1

    if l1 < dim:
        wide_x = int(math.ceil(dim * 1.0 / l1))

    if l2 < dim:
        wide_y = int(math.ceil(dim * 1.0 / l2))

    Mat = np.zeros(np.shape(mat))
    Mat = mat
    for i in range(1, wide_x):
        Mat = np.append(Mat, mat, axis=0)

    Matt = np.zeros(np.shape(Mat))
    Matt = Mat
    for i in range(1, wide_y):
        Matt = np.append(Matt, Mat, axis=1)

    dim1 = np.shape(Matt)[0] * 1.0
    dim2 = np.shape(Matt)[1] * 1.0

    chunk_size1 = int(dim1 / dim)
    chunk_size2 = int(dim2 / dim)

    pooled_mat = np.zeros((dim, dim))

    for i in range(dim * chunk_size1, int(dim1)):
        for j in range(0, int(dim2)):
            if type == 'max':
                Matt[i - (int(dim1) - dim * chunk_size1)][j] = max(Matt[i - (int(dim1) - dim * chunk_size1)][j],
                                                                   Matt[i][j])
            elif type == 'min':
                Matt[i - (int(dim1) - dim * chunk_size1)][j] = min(Matt[i - (int(dim1) - dim * chunk_size1)][j],
                                                                   Matt[i][j])

    for j in range(dim * chunk_size2, int(dim2)):
        for i in range(0, int(dim1)):
            if type == 'max':
                Matt[i][j - (int(dim2) - dim * chunk_size2)] = max(Matt[i][j - (int(dim2) - dim * chunk_size2)],
                                                                   Matt[i][j])
            elif type == 'min':
                Matt[i][j - (int(dim2) - dim * chunk_size2)] = min(Matt[i][j - (int(dim2) - dim * chunk_size2)],
                                                                   Matt[i][j])

    for i in range(0, dim):
        for j in range(0, dim):
            val = Matt[i * chunk_size1][j * chunk_size2]
            for u in range(i * chunk_size1, (i + 1) * chunk_size1):
                for v in range(j * chunk_size2, (j + 1) * chunk_size2):
                    if type == 'max':
                        val = max(val, Matt[u][v])
                    elif type == 'min':
                        val = min(val, Matt[u][v])

            pooled_mat[i][j] = val

    return pooled_mat


def getFlattenedDataforClassifier(data, De, En, stopWordsRemoval=False):
    X = []
    Y = []
    for e in data:
        e[0] = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in e[0].split()]
        e[1] = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in e[1].split()]

        vec1 = []
        for word in e[0]:
            try:
                vec1.append(De[word])
            except:
                print word + " not in dictionary"

        if (len(vec1) == 0):
            vec1.append(De['.'])

        vec2 = []

        for word in e[1]:
            try:
                vec2.append(En[word])
            except:
                print word + " not in dictionary"

        if (len(vec2) == 0):
            vec2.append(En['.'])

        print len(vec1), len(vec2)

        mat = np.ndarray.flatten(getDynamicPooledMatrix(getSimilarityMatrix(vec1, vec2)))
        X.append(mat)
        Y.append(e[2])
    return np.array(X), np.array(Y)
