import numpy as np 

import sklearn.metrics as sk
import math
import bilingual
import BUCC
import pickle
import codecs



def getSimilarityMatrix(vec1, vec2,metric='cos'):

	mat = np.zeros((len(vec1),len(vec2)))

	for i in range(0,len(vec1)):
		for j in range(0,len(vec2)):
			if(metric == 'cos'):
				mat[i][j] = sk.pairwise.cosine_similarity(vec1[i],vec2[j])
			elif metric == 'euclid':
				mat[i][j] = sk.pairwise.euclidean_distances(vec1[i],vec2[j])

	return mat


def getDynamicPooledMatrix(mat,dim=15,type='max'):

	l1 = np.shape(mat)[0]
	l2 = np.shape(mat)[1]


	wide_x =  wide_y = 1
	
	if l1 < dim:
		wide_x = int(math.ceil(dim*1.0/l1))
		#print wide_x

	if l2 < dim:
		wide_y = int(math.ceil(dim*1.0/l2))
		#print wide_y


	Mat = np.zeros(np.shape(mat))
	Mat = mat
	for i in range(1, wide_x):
		Mat = np.append(Mat,mat,axis=0)

	Matt = np.zeros(np.shape(Mat))
	Matt = Mat
	for i in range(1,wide_y):
		Matt = np.append(Matt,Mat,axis=1)

	#print np.shape(Matt)
	#print Matt

	### NOW we have large matrix start pooling now
	##### chunk_size1 =  int(dim1/dim)
	##### chunk_size2 = int(dim2/dim)
	# stride over all the chunks one by one

	dim1 = np.shape(Matt)[0]*1.0
	dim2 = np.shape(Matt)[1]*1.0

	chunk_size1 = int(dim1/dim)
	chunk_size2 = int(dim2/dim)

	pooled_mat =  np.zeros((dim,dim))

	for i in range(dim*chunk_size1,int(dim1)):
		for j in range(0,int(dim2)):
			if type == 'max':
				Matt[i - (int(dim1)-dim*chunk_size1)][j] =  max(Matt[i - (int(dim1)-dim*chunk_size1)][j],Matt[i][j])
			elif type == 'min':
				Matt[i - (int(dim1)-dim*chunk_size1)][j] =  min(Matt[i - (int(dim1)-dim*chunk_size1)][j],Matt[i][j])



	for j in range(dim*chunk_size2,int(dim2)):
		for i in range(0, int(dim1)):
			if type == 'max':
				Matt[i][j - (int(dim2)-dim*chunk_size2)] =  max(Matt[i][j - (int(dim2)-dim*chunk_size2)],Matt[i][j])
			elif type == 'min':
				Matt[i][j - (int(dim2)-dim*chunk_size2)] =  min(Matt[i][j - (int(dim2)-dim*chunk_size2)],Matt[i][j])

	#print Matt

	for i in range(0,dim):
		for j in range(0,dim):
			val = Matt[i*chunk_size1][j*chunk_size2]
			for u in range(i*chunk_size1,(i+1)*chunk_size1):
				for v in range(j*chunk_size2,(j+1)*chunk_size2):
					if type == 'max':
						val =  max(val,Matt[u][v])
					elif type == 'min':
						val = min(val,Matt[u][v])

			pooled_mat[i][j] = val

	#print pooled_mat

	return pooled_mat



def getFlattenedDataforClassifier(data,De,En,stopWordsRemoval=False):

	X = []
	Y=  []

	for e in data:
		#print e[0].split()
		#print e[1].split()
		#print e[2]
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

		mat = np.ndarray.flatten(getDynamicPooledMatrix(getSimilarityMatrix(vec1,vec2)))
		X.append(mat)
		Y.append(e[2])
		

		

	return np.array(X), np.array(Y)





def get_Word_Vecs_for_Data(stopWordsRemoval=False):
	De =  bilingual.loadWordvecs('../data/unsup.128.de')
	En  = bilingual.loadWordvecs('../data/unsup.128.en')
	train_data, valid_data, test_data  = BUCC.loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')

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

	filed = open('../data/data/bucc2017/de-en/train_valid_test/data_de_en_vecs','rb')
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
	De =  bilingual.loadWordvecs('../data/unsup.128.de')
	En  = bilingual.loadWordvecs('../data/unsup.128.en')
	train_data, valid_data, test_data  = BUCC.loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')

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
		#print e[0].split()
		#print e[1].split()
		#print e[2]
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
	filed = open('../data/data/bucc2017/de-en/train_valid_test/data_de_en_simMat_without_dp','rb')
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
	De =  bilingual.loadWordvecs('../data/unsup.128.de')
	En  = bilingual.loadWordvecs('../data/unsup.128.en')
	train_data, valid_data, test_data  = BUCC.loadTrainTestValidationSampledData('../data/data/bucc2017/de-en/train_valid_test/data')

	train_X, train_Y = getFlattenedDataforBucketedClassifier(train_data,De,En)
	valid_X, valid_Y = getFlattenedDataforBucketedClassifier(valid_data,De,En)
	test_X, test_Y = getFlattenedDataforBucketedClassifier(test_data, De, En)



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
		#print e[0].split()
		#print e[1].split()
		#print e[2]
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

	filed = open('../data/data/bucc2017/de-en/train_valid_test/data_de_en_vecs_bucketed','rb')
	data = pickle.load(filed)
	filed.close()

	train_X = data[0]
	train_Y = data[1]
	valid_X = data[2]
	valid_Y = data[3]
	test_X  = data[4]
	test_Y  = data[5]

	return train_X, train_Y, valid_X, valid_Y,test_X,test_Y

	
	









# def train_Classifier():
# 	import tensorflow as tf



# 	no_of_hidden_units = 200
# 	no_of_epochs = 10
# 	batchSize = 5
# 	eta = 100

# 	hyper = 'simple_dynamic_pool_dense_'+str(no_of_hidden_units)+'_'+str(no_of_epochs)+'_'+str(batchSize)+'_'+str(eta)



################################### CLASSIFIER ##################
import tensorflow as tf
no_of_hidden_units = 100
no_of_epochs = 100
batchSize = 1
eta = 1

hyper = 'simple_dynamic_pool_dense_'+str(no_of_hidden_units)+'_'+str(no_of_epochs)+'_'+str(batchSize)+'_'+str(eta)

def train(trainX, trainY):
    '''
    Complete this function.
    '''

    perm = np.random.permutation(trainY.shape[0])
    trainX = trainX[perm]
    trainY = trainY[perm]

    

    ################# NETWORK DIMENSIONS #####################
    no_of_inputs = np.shape(trainX)[0]
    print np.shape(trainX)
    input_size = len(np.ndarray.flatten(np.array(trainX[0],copy=True)))
    print input_size
    output_size = 1
    
    #trainX = [ np.ndarray.flatten(x)  for x in trainX]
    trainY = np.array([np.array([x]) for x in trainY])
    print np.shape(trainY)

    # trainYY = []
    # for i in range(0,len(trainY)):
    #     label = trainY[i]
    #     trainYY.append(np.zeros(output_size))
    #     trainYY[i][label] = 1



    ############### MODEL ##################

    W1 = tf.Variable(tf.random_normal([input_size, no_of_hidden_units] ,-1), name = 'W1')*(1/np.sqrt(no_of_hidden_units*input_size))
    b1 = tf.Variable(tf.zeros([no_of_hidden_units]) , name = 'b1')

    W2 = tf.Variable(tf.random_normal([ no_of_hidden_units, output_size] ,-1), name = 'W2')*(1/np.sqrt(output_size*no_of_hidden_units))
    b2 = tf.Variable(tf.zeros([output_size]) , name = 'b2')

    x = tf.placeholder(tf.float32, [None,input_size])
    y = tf.placeholder(tf.float32, [None,output_size])

    z1 = tf.matmul(x,W1) + b1
    a1 = tf.nn.sigmoid(z1)

    z2 = tf.matmul(a1,W2) + b2
    a2 = tf.nn.sigmoid(z2)

    #loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,a2)),1))
    loss = tf.reduce_mean(tf.square(tf.sub(y,a2)))

    train_step = tf.train.GradientDescentOptimizer(eta).minimize(loss)

    
    sess  = tf.InteractiveSession()

    tf.initialize_all_variables().run()


    for e in range(0,no_of_epochs):

        for j in xrange(0,no_of_inputs,batchSize):
            batchlen = min(batchSize,no_of_inputs-j)

            X = trainX[j:j+batchlen]
            Y = trainY[j:j+batchlen]
            # print X
            # print Y

            sess.run(train_step,feed_dict={x : X,y : Y })
            print (sess.run([loss],feed_dict={x : X,y : Y }))



    np.save('weights/w1'+hyper,W1.eval())
    np.save('weights/w2'+hyper,W2.eval())
    np.save('weights/b1'+hyper,b1.eval())
    np.save('weights/b2'+hyper,b2.eval())








def test(testX,test_Y,th=0.5):
    '''
    Complete this function.
    This function must read the weight files and
    return the predicted labels.
    The returned object must be a 1-dimensional numpy array of
    length equal to the number of examples. The i-th element
    of the array should contain the label of the i-th test
    example.
    '''

    testX = [ np.ndarray.flatten(x)  for x in testX]
    input_size = len(np.ndarray.flatten(np.array(testX[0],copy=True)))

    w1 = np.load('weights/w1'+hyper+'.npy')
    bb1 = np.load('weights/b1'+hyper+'.npy')
    w2 = np.load('weights/w2'+hyper+'.npy')
    bb2 = np.load('weights/b2'+hyper+'.npy')

    print "here3"

    W1 = tf.constant(w1)#tf.Variable(tf.truncated_normal([input_size, no_of_hidden_units] ,1), name = 'W1')*(1/np.sqrt(input_size*no_of_hidden_units))
    b1 = tf.constant(bb1)#tf.Variable(tf.zeros([no_of_hidden_units]) , name = 'b1')

    print "hre 5"
    W2 = tf.constant(w2)#tf.Variable(tf.truncated_normal([ no_of_hidden_units, output_size] ,1), name = 'W2')*(1/np.sqrt(output_size*no_of_hidden_units))
    b2 = tf.constant(bb2)#tf.Variable(tf.zeros([output_size]) , name = 'b2')
    print "hre 5"
    x = tf.placeholder(tf.float32, [None,input_size])
    #y = tf.placeholder(tf.float32, [None,output_size])
    print "hre 5"
    z1 = tf.matmul(x,W1) + b1
    a1 = tf.nn.sigmoid(z1)
    print "hre 5"
    z2 = tf.matmul(a1,W2) + b2
    a2 = tf.nn.sigmoid(z2)
    print "hre 7"
    sess  = tf.InteractiveSession()
    print "hre 5"
    tf.initialize_all_variables().run()

    print "here4"


    val = []
    for i in xrange(0,len(testX),batchSize):
    	batchlen = min(batchSize,len(testX)-i)
    	vals = sess.run(a2,feed_dict={x:testX[i:i+batchlen]})
    	val.extend(vals)
	    #print val[0]

    ans = np.array([1 if x>th else 0 for x in val])
    print ans[0]
    print np.shape(ans)
    print np.shape(testX)
    val = np.mean((ans == test_Y))*100.0

    for i in range(0,20):
		print ans[i],test_Y[i]

    print "here1"

    print val
    print sk.recall_score(test_Y,ans)
    print sk.precision_score(test_Y,ans)


    cnt = 0
    cntT = 0
    for i in range(0,len(test_Y)):
	    if ans[i]== 1:
		    cnt  += 1
		    if test_Y[i] == 1:
			    cntT += 1

    print cnt, cntT
    print "here2"

    P1A1 = P0A0 = P1A0 = P0A1 = 0
    for i in range(0,len(test_Y)):
    	if ans[i] == 1:
    		if test_Y[i] == 1:
    			P1A1 += 1
    		else:
    			P1A0 += 1
    	else:
    		if test_Y[i] == 1:
    			P0A1 += 1
    		else:
    			P0A0 += 1

    with open("results/"+'classsifier_with_stopwords_results', "a") as text_file:
        print >> text_file, hyper
        print >> text_file, P1A1, P1A0, P0A1,P0A0
        print >> text_file ,'Accururacy : '+ str(val)
        print >> text_file ,'Precision : '+ str(sk.precision_score(test_Y,ans))
        print >> text_file, 'Recall : '+ str(sk.recall_score(test_Y,ans))
        print >> text_file, 'F1 : '+ str(sk.f1_score(test_Y,ans))
        print >>text_file, '\n'


    return ans


def getDataForTranslationUsingSeq2SeqModels(file_data,ind=0):
	train_data, valid_data, test_data  = BUCC.loadTrainTestValidationSampledData(file_data)

	name = file_data.split('/')[-1]

	f = codecs.open('seq2seq/data/' +name +'.txt', 'w',encoding='utf-8' )
	for i in train_data:
		print >> f,  i[ind].strip('.,?!')
		#print >> f,  i[ind+1].strip('.,?!')
		#print >> f,  i[ind+2]


	for i in valid_data:
		print >> f,  i[ind].strip('.,?!')

	for i in test_data:
		print >> f,  i[ind].strip('.,?!')






	
def main():
	#getDynamicPooledMatrix([[1,2,3],[4,5,6]])

	######################### CODE FOR  MATRIX SIMILARITY #########################
	#get_Word_Vecs_for_Data()
	train_X, train_Y, valid_X, valid_Y,test_X,test_Y = load_Word_Vecs_for_Data()
	# train(train_X,train_Y)

	# print "hogya\n\n\n"
	# print len(test_X)
	
	ans  = test(test_X,test_Y)

	####### Cdoe for getting translations and then running the model to train them ######
	#getDataForTranslationUsingSeq2SeqModels('../data/data/bucc2017/de-en/train_valid_test/data')

	

	


if __name__ == "__main__":
	main()

