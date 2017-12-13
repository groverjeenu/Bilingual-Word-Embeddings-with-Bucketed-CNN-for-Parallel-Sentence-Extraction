import numpy as np 
import sklearn.metrics as sk
import math
import utils
import config
import BUCCDataHandler
import pickle
import codecs
import DynamicPooling
import os


getSimilarityMatrix = DynamicPooling.getSimilarityMatrix
getDynamicPooledMatrix = DynamicPooling.getDynamicPooledMatrix
getFlattenedDataforClassifier = DynamicPooling.getFlattenedDataforClassifier
dir = os.path.dirname(__file__)


def get_Word_Vecs_for_Data(stopWordsRemoval=False):
	De =  utils.loadWordvecs('../data/unsup.128.de')
	En =  utils.loadWordvecs('../data/unsup.128.en')
	train_data, valid_data, test_data  = BUCCDataHandler.loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')

	train_X, train_Y = getFlattenedDataforClassifier(train_data,De,En)
	valid_X, valid_Y = getFlattenedDataforClassifier(valid_data,De,En)
	test_X, test_Y = getFlattenedDataforClassifier(test_data, De, En)


	print np.shape(train_X)
	print np.shape(train_Y)


	dump = []
	dump.append(train_X)
	dump.append(train_Y)
	dump.append(valid_X)
	dump.append(valid_Y)
	dump.append(test_X)
	dump.append(test_Y)

	filed = open('../data/data/bucc2017/de-en/train_valid_test/data_de_en_vecs','wb')
	pickle.dump(dump,filed)
	filed.close()


	return train_X,train_Y,valid_X,valid_Y,test_X,test_Y

def load_Word_Vecs_for_Data(stopWordsRemoval=False):

	filed = open(os.path.join(dir,'../data/data/bucc2017/de-en/train_valid_test/data_de_en_vecs'),'rb')
	data = pickle.load(filed)
	filed.close()

	train_X = data[0]
	train_Y = data[1]
	valid_X = data[2]
	valid_Y = data[3]
	test_X  = data[4]
	test_Y  = data[5]

	return train_X, train_Y, valid_X, valid_Y,test_X,test_Y



def get_data_for_plotting():
	De =  utils.loadWordvecs('../data/unsup.128.de')
	En  = utils.loadWordvecs('../data/unsup.128.en')
	train_data, valid_data, test_data  = BUCCDataHandler.loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')

	train_X, train_Y = getSimMatforPlotting(train_data,De,En)
	valid_X, valid_Y = getSimMatforPlotting(valid_data,De,En)
	test_X, test_Y = getSimMatforPlotting(test_data, De, En)


	print np.shape(train_X)
	print np.shape(train_Y)


	dump = []
	dump.append(train_X)
	dump.append(train_Y)
	dump.append(valid_X)
	dump.append(valid_Y)
	dump.append(test_X)
	dump.append(test_Y)

	filed = open('../data/data/bucc2017/de-en/train_valid_test/data_de_en_simMat_without_dp','wb')
	pickle.dump(dump,filed)
	filed.close()


	return train_X,train_Y,valid_X,valid_Y,test_X,test_Y


def getSimMatforPlotting(data,De,En,stopWordsRemoval=False):

	X = []
	Y=  []

	for e in data:
		e[0] = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in e[0].split()]
		e[1] = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in e[1].split()]

		vec1 = []
		for word in e[0]:
			try:
				vec1.append(De[word])
			except:
				print word+" not in dictionary"

		if(len(vec1) == 0):
			vec1.append(De['.'])

		vec2 = []

		for word in e[1]:
			try:
				vec2.append(En[word])
			except:
				print word+" not in dictionary"

		if(len(vec2) == 0):
			vec2.append(En['.'])

		print len(vec1), len(vec2)

		mat = getSimilarityMatrix(vec1,vec2)
		X.append(mat)
		Y.append(e[2])

	return np.array(X), np.array(Y)

def load_data_for_plotting(stopWordsRemoval=False):
	filed = open(os.path.join(dir,'../data/data/bucc2017/de-en/train_valid_test/data_de_en_simMat_without_dp'),'rb')
	data = pickle.load(filed)
	filed.close()

	train_X = data[0]
	train_Y = data[1]
	valid_X = data[2]
	valid_Y = data[3]
	test_X  = data[4]
	test_Y  = data[5]

	return train_X, train_Y, valid_X, valid_Y,test_X,test_Y



def get_Word_Vecs_for_Data_Bucketed(buckets,stopWordsRemoval=False):
	De =  utils.loadWordvecs('../data/unsup.128.de')
	En  = utils.loadWordvecs('../data/unsup.128.en')
	train_data, valid_data, test_data  = BUCCDataHandler.loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')

	train_X, train_Y = getFlattenedDataforBucketedClassifier(train_data,De,En,buckets)
	valid_X, valid_Y = getFlattenedDataforBucketedClassifier(valid_data,De,En,buckets)
	test_X, test_Y = getFlattenedDataforBucketedClassifier(test_data, De, En,buckets)

	print np.shape(train_X)
	print np.shape(train_Y)

	dump = []
	dump.append(train_X)
	dump.append(train_Y)
	dump.append(valid_X)
	dump.append(valid_Y)
	dump.append(test_X)
	dump.append(test_Y)

	filed = open('../data/data/bucc2017/de-en/train_valid_test/data_de_en_vecs_bucketed','wb')
	pickle.dump(dump,filed)
	filed.close()


	return train_X,train_Y,valid_X,valid_Y,test_X,test_Y


def getFlattenedDataforBucketedClassifier(data,De,En,buckets,stopWordsRemoval=False):

	X = []
	Y=  []

	for i in buckets:
		X.append([])
		Y.append([])

	for e in data:
		e[0] = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in e[0].split()]
		e[1] = [f.strip('.,:"!@#$%^&*(){}{}?').lower() for f in e[1].split()]

		vec1 = []
		for word in e[0]:
			try:
				vec1.append(De[word])
			except:
				print word+" not in dictionary"

		if(len(vec1) == 0):
			vec1.append(De['.'])

		vec2 = []

		for word in e[1]:
			try:
				vec2.append(En[word])
			except:
				print word+" not in dictionary"

		if(len(vec2) == 0):
			vec2.append(En['.'])


		print len(vec1), len(vec2)

		l = (len(vec1) + len(vec2))/2.0

		bucket = -1
		bucketind = -1
		for ind,val in enumerate(buckets):
			if l <= val:
				bucket = val
				bucketind = ind
				break

		if bucket == -1:
			bucket = buckets[-1]
			bucketind = len(buckets)-1

		print "bucket is : ",bucket
		print "ind is : ",bucketind
		mat = np.ndarray.flatten(getDynamicPooledMatrix(getSimilarityMatrix(vec1,vec2),dim=bucket))
		X[bucketind].append(mat)
		Y[bucketind].append(e[2])
		

		

	return np.array(X), np.array(Y)


def load_Word_Vecs_for_Data_Bucket(buckets,stopWordsRemoval=False):

	filed = open(os.path.join(dir,'../data/data/bucc2017/de-en/train_valid_test/data_de_en_vecs_bucketed'),'rb')
	data = pickle.load(filed)
	filed.close()

	train_X = data[0]
	train_Y = data[1]
	valid_X = data[2]
	valid_Y = data[3]
	test_X  = data[4]
	test_Y  = data[5]

	return train_X, train_Y, valid_X, valid_Y,test_X,test_Y

	
	
def main():
	# Run this only to regenerate matrices
	get_Word_Vecs_for_Data()
	get_Word_Vecs_for_Data_Bucketed(config.buckets)
	get_data_for_plotting()
	



if __name__ == "__main__":
	main()

