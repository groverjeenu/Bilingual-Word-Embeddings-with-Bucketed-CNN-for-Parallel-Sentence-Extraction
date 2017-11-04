
import numpy as np
import random

import codecs

import pickle

def getBUCCdataforRAE(file1, file2, fileg):
	f = open(fileg,'r')
	Gold = []
	for line in f:
		line  = line.split()
		Gold.append(line)

	l = len(Gold)
	random.shuffle(Gold)

	validation_gold = Gold[int(0.6*l):]

	f = codecs.open(file1,'r',encoding='utf-8')
	#f = codecs.reader(f)
	for line in f:
		line = line[line.find('\t')+1:].strip('\n')
		print line
		#break


def random_bin(p):
	l = np.random.randint(0,1000)

	if l <= int(1000*p):
		#print 1
		return 1 
	#print 0
	return 0


def samplePositiveNegative(gold, Gold_dict,F1,F2,file1, file2):  ###### File1 must be the first column in gold
	p = 0.05

	Data = []   ############ GET ALL POSITIVES
	for g in gold:
		d = []
		d.append(F1[g[0]])
		d.append(F2[g[1]])
		d.append(1)

		Data.append(d)

	######### NAIVE NEgative Sampling


	list_F2 = F2.keys()
	len_F2 = len(list_F2)
	for ind,j in enumerate(F1.keys()):
		#print ind, j
		print len(Data)
		if random_bin(p*(len(gold))/len(Gold_dict.keys())):
			d=[]
			val = list_F2[np.random.randint(0,len_F2)]
			try:
				if Gold_dict[j] !=  val:
					#print "here1"
					d.append(F1[j])
					d.append(F2[val])
					d.append(0)
					Data.append(d)
			except:
				#print "here2"
				d.append(F1[j])
				d.append(F2[val])
				d.append(0)
				Data.append(d)

	random.shuffle(Data)
	return Data


def  getTrainTestValidationSampledData(file1, file2 ,fileg):

	train_split = 0.6
	valid_split = 0.1
	test_split  = 0.3

	f = open(fileg,'r')
	Gold = []

	Gold_dict = {}
	for line in f:
		line  = line.split()
		Gold_dict[line[0]] = line[1]
		Gold.append(line)

	random.shuffle(Gold)
	l=  len(Gold)

	f = codecs.open(file1,'r',encoding='utf-8')

	F1 = {}
	for line in f:
		ind  =  line.find('\t')
		sent = line[ind+1:].strip('\n')
		id_ = line[:ind]
		# for j in line:
		# 	print j
		F1[id_] = sent


	f = codecs.open(file2,'r',encoding='utf-8')

	F2 = {}
	for line in f:
		ind  =  line.find('\t')
		sent = line[ind+1:].strip('\n')
		id_ = line[:ind]
		# for j in line:
		# 	print j
		F2[id_] = sent

	print len(F1.keys())
	print len(F2.keys())

	train_gold = Gold[:int(train_split*l)]
	valid_gold = Gold[int(train_split*l) : int((train_split+valid_split)*l)]
	test_gold = Gold[int((train_split+valid_split)*l) : ]


	

	train_data = samplePositiveNegative(train_gold,Gold_dict,F1,F2,file1,file2)
	valid_data = samplePositiveNegative(valid_gold,Gold_dict,F1,F2,file1,file2)
	test_data = samplePositiveNegative(test_gold,Gold_dict,F1,F2,file1,file2)

	for i in range(0,10):
		print train_data[i]

	dump = []
	dump.append(train_data)
	dump.append(valid_data)
	dump.append(test_data)


	filed = open('../data/data/bucc2017/de-en/train_valid_test/data','wb')
	pickle.dump(dump,filed)
	filed.close()


	return train_data,valid_data,test_data


def loadTrainTestValidationSampledData(file1):
	filed = open(file1,'rb')
	data = pickle.load(filed)
	filed.close()

	train_data,valid_data,test_data = data[0],data[1],data[2]

	return train_data,valid_data,test_data




def getAllDataForTrainingRAE(file,n=10):
	########## GENATED DATA FOR TRAINING RAE
	####### SAMPLES 1/n th fraction of sentnecs from BUCC Monoligua Data

	f = codecs.open(file,'r',encoding='utf-8')
	f1 = codecs.open(file+'.RAE','w',encoding='utf-8')
	#f = codecs.reader(f)

	cnt  = 0
	for line in f:
		cnt = cnt +1
		line = line[line.find('\t')+1:].strip('\n')
		# for j in line:
		# 	print j
		if cnt %n == 0:
			print >> f1, line

def getAllDataForTrainingBivec(file,n=10):
	########## GENATED DATA FOR TRAINING RAE
	####### SAMPLES 1/n th fraction of sentnecs from BUCC Monoligua Data

	f = codecs.open(file,'r',encoding='utf-8')
	#f1 = codecs.open(file+'.bivec','w',encoding='utf-8')
	cnt = 0
	for line in f:
		# line = line.strip('\n').lower()
		# print >> f1, line
		cnt = cnt+1
	print cnt


def getTrainValidTestDataforTrainingTreeLSTM(file):
	f = codecs.open(file,'r',encoding='utf-8')
	f1 = codecs.open(file+'.train','w',encoding='utf-8')
	f2 = codecs.open(file+'.valid','w',encoding='utf-8')
	f3 = codecs.open(file+'.test','w',encoding='utf-8')

	cnt  = 0
	for line in f:
		cnt = cnt +1
	f.close()

	f = codecs.open(file,'r',encoding='utf-8')

	Train_end  = 0.65
	Valid_end = 0.75


	val = 0
	for line in f:
		val =  val + 1
		line = line[line.find('\t')+1:].strip('\n.;;?]{()}!@#%^&*<>,[').lower()

		if(val <= int(Train_end*cnt)):
			print >> f1, line
		elif val <= int(Valid_end*cnt):
			print >> f2, line
		else:
			print >> f3, line











		



def main():
	#getBUCCdataforRAE('../data/data/bucc2017/de-en/de-en.training.de','d','../data/data/bucc2017/de-en/de-en.training.gold')
	

	# getTrainTestValidationSampledData('../data/data/bucc2017/de-en/de-en.training.de','../data/data/bucc2017/de-en/de-en.training.en','../data/data/bucc2017/de-en/de-en.training.gold')
	# #getAllDataForTrainingRAE('../data/data/bucc2017/de-en/de-en.training.en')
	# train_data,valid_data,test_data = loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')#
	# print len(train_data)
	# print len(valid_data)
	# print len(test_data)
	# #samplePositiveNegative([],'../data/data/bucc2017/de-en/de-en.training.en','../data/data/bucc2017/de-en/de-en.training.de')

	#getTrainValidTestDataforTrainingTreeLSTM('../data/data/bucc2017/de-en/de-en.training.de')

	#getAllDataForTrainingBivec('../data/hansards.fr')

	x,y,z = loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')
	print len(x)
	print x[0]
	print len(y)
	print y[0]
	print len(z)
	print z[0]








if __name__ == "__main__":
	main()
