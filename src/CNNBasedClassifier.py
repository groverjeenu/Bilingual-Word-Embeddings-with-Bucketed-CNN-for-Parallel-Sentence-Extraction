import numpy as np
import sklearn.metrics as sk


import numpy as np
import tensorflow as tf

##################### MODEL PARAMETERS and HYPERPARAMETERS DEFINED #####################
input_height = input_width = 28 ########## INPUT DIMENSIONS  ######
input_channels = 1
output_size =  1  ###### OUTPUT DIMENSIONS ##########

filter_1_height = filter_1_width = 3 ######### FILTER 1 #####
filter_1_number = 12
filter_2_height = filter_2_width = 3 ######## FILTER 2 #####
filter_2_number = 16

pool_1_size  = [1,2,2,1]  ###### POOLING LAYER 1 PARAMTERS ######
pool_1_stride = [1,2,2,1]

pool_2_size = [1,2,2,1]  ###### POOLING LAYER 2 PARAMTERS ######
pool_2_stride = [1,2,2,1]

dropout_prob  = 0.2    ############### DROPOUT PROBABLILTY ######

no_of_hidden_units = 200

no_of_epochs = 20
batchSize = 5
eta = 0.0005  #10,25,0.0005 -> 98.29
lamda = 1     ####### REGULARIZATION PARAMETER ###########
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
hyper = 'sntSim_cnn_'+str(no_of_hidden_units)+'_'+str(no_of_epochs)+'_'+str(batchSize)+'_'+str(eta)
resultsFile = 'classsifier_CNN_results'



def set_global_var(ind,dim,bucketed=False):
	global input_height
	global input_width
	global filter_1_number
	global filter_2_number
	global no_of_hidden_units
	global no_of_epochs
	global dropout_prob
	global batchSize
	global eta
	global hyper
	global resultsFile
	input_width = input_height = dim

	s = '_sentSim_cnn_'
	if not bucketed:
		s = s + 'Nonbucketed_'
		resultsFile = resultsFile + '_Nonbucketed'

	hyper = s+str(dim)+'_'+str(no_of_hidden_units)+'_'+str(no_of_epochs)+'_'+str(batchSize)+'_'+str(eta)





def train(trainX, trainY):  ################ FUNCT TO TRAIN A CNN ##########################
    no_of_inputs = len(trainX)
    print no_of_inputs
    print np.shape(trainX[0])

    trainXX = []
    for ind,x in enumerate(trainX):
    	trainXX.append(np.reshape(x,(input_width,input_height,input_channels)))

    trainX = trainXX


    ############## Convert Y to ONE-HOT encoding ##########
    trainYY = []
    for i in range(0,len(trainY)):
        label = trainY[i]
        trainYY.append(np.zeros(output_size))
        trainYY[i][0] = label

    ########### PLACEHODERS DEFINED #######################
    x = tf.placeholder(tf.float32, [None,input_height,input_width, input_channels])
    y = tf.placeholder(tf.float32, [None,output_size])


    ############### DECLARE CONVULTION VARIABLES #######################
    W_conv1_shape = [filter_1_height,filter_1_width,input_channels,filter_1_number]
    W_conv1 = tf.Variable(tf.truncated_normal(W_conv1_shape,stddev=0.1))
    b_conv1_shape = [filter_1_number]
    b_conv1 =  tf.Variable(tf.zeros(b_conv1_shape))


    W_conv2_shape = [filter_2_height,filter_2_width,filter_1_number,filter_2_number]
    W_conv2 = tf.Variable(tf.truncated_normal(W_conv2_shape,stddev=0.1))
    b_conv2_shape = [filter_2_number]
    b_conv2 = tf.Variable(tf.zeros(b_conv2_shape))


    ########## MAKE CONNECTIONS of ConV Layers ###################
    W_conv1_z = tf.nn.conv2d(x,W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    W_conv1_h = tf.nn.relu(W_conv1_z)
    pool_1_h = tf.nn.max_pool(W_conv1_h, ksize=pool_1_size,strides=pool_1_stride, padding='SAME')

    W_conv2_z = tf.nn.conv2d(pool_1_h,W_conv2,strides=[1,1,1,1],padding='SAME') + b_conv2
    W_conv2_h = tf.nn.relu(W_conv2_z)
    pool_2_h = tf.nn.max_pool(W_conv2_h,ksize=pool_2_size,strides=pool_2_stride,padding='SAME')


    ################# GET FEATURES FOR FULLY CONNECTED LAYERS ###############
    no_of_features = pool_2_h.get_shape()[1:4].num_elements()
    input_FC1 = tf.reshape(pool_2_h , [-1,no_of_features])

    ############# PARAMETERS FOR FC LAYERS ######################
    W1 = tf.Variable(tf.truncated_normal([no_of_features,no_of_hidden_units],stddev=0.1))
    b1 = tf.Variable(tf.zeros([no_of_hidden_units]))

    W2 = tf.Variable(tf.truncated_normal([no_of_hidden_units,output_size],stddev=0.1))
    b2 = tf.Variable(tf.zeros([output_size]))

    ############ Connect FC LAYERS ############################

    z1 = tf.matmul(input_FC1,W1) + b1
    h1 = tf.nn.relu(z1)

    ########## DROPOUT #########################
    prob = tf.placeholder(tf.float32)
    out_drop = tf.nn.dropout(h1,prob)

    z2 = tf.matmul(out_drop,W2) + b2
    h2 = tf.sigmoid(z2)


    ##################### TRAINING and LOSS ###################
    loss = tf.reduce_mean(tf.square(y-h2))
    train_step = tf.train.AdamOptimizer(eta).minimize(loss)

    sess  = tf.InteractiveSession()
    tf.initialize_all_variables().run()
    for e in range(0,no_of_epochs):

        for j in xrange(0,no_of_inputs,batchSize):  ####### TRAIN A BATCH #############
            batchlen = min(batchSize,no_of_inputs-j)

            X = trainX[j:j+batchlen]
            Y = trainYY[j:j+batchlen]
            
            sess.run(train_step,feed_dict={x : X,y : Y,prob : dropout_prob })
            print (sess.run([loss],feed_dict={x : X,y : Y, prob : dropout_prob }))

    ############################# SAVE WEIGHTS #########################
    np.save('weights/w1'+hyper,W1.eval())
    np.save('weights/w2'+hyper,W2.eval())
    np.save('weights/b1'+hyper,b1.eval())
    np.save('weights/b2'+hyper,b2.eval())
    np.save('weights/W_conv1'+hyper,W_conv1.eval())
    np.save('weights/W_conv2'+hyper,W_conv2.eval())
    np.save('weights/b_conv1'+hyper,b_conv1.eval())
    np.save('weights/b_conv2'+hyper,b_conv2.eval())

    print "Training of CNN Network completed\n"



def test(testX,test_Y,th=0.5):
    W1_val = np.load('weights/w1'+hyper+'.npy')
    W2_val = np.load('weights/w2'+hyper+'.npy')
    b1_val = np.load('weights/b1'+hyper+'.npy')
    b2_val = np.load('weights/b2'+hyper+'.npy')
    W_conv1_val = np.load('weights/W_conv1'+hyper+'.npy')
    W_conv2_val = np.load('weights/W_conv2'+hyper+'.npy')
    b_conv1_val = np.load('weights/b_conv1'+hyper+'.npy')
    b_conv2_val = np.load('weights/b_conv2'+hyper+'.npy')

    testXX= []
    for ind,x in enumerate(testX):
    	testXX.append(np.reshape(x,(input_width,input_height,input_channels)))
    testX = testXX


    ########### PLACEHODERS DEFINED #######################
    x = tf.placeholder(tf.float32, [None,input_height,input_width, input_channels])
    

    ############### DECLARE CONVULTION VARIABLES #######################
    W_conv1 = tf.Variable(W_conv1_val)
    b_conv1 =  tf.Variable(b_conv1_val)

    W_conv2 = tf.Variable(W_conv2_val)
    b_conv2 = tf.Variable(b_conv2_val)

    ########## MAKE CONNECTIONS of ConV Layers ###################
    W_conv1_z = tf.nn.conv2d(x,W_conv1,strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    W_conv1_h = tf.nn.relu(W_conv1_z)
    pool_1_h = tf.nn.max_pool(W_conv1_h, ksize=pool_1_size,strides=pool_1_stride, padding='SAME')

    W_conv2_z = tf.nn.conv2d(pool_1_h,W_conv2,strides=[1,1,1,1],padding='SAME') + b_conv2
    W_conv2_h = tf.nn.relu(W_conv2_z)
    pool_2_h = tf.nn.max_pool(W_conv2_h,ksize=pool_2_size,strides=pool_2_stride,padding='SAME')

    ################# GET FEATURES FOR FULLY CONNECTED LAYERS ###############
    no_of_features = pool_2_h.get_shape()[1:4].num_elements()
    input_FC1 = tf.reshape(pool_2_h , [-1,no_of_features])

    ############# PARAMETERS FOR FC LAYERS ######################
    W1 = tf.Variable(W1_val)
    b1 = tf.Variable(b1_val)

    W2 = tf.Variable(W2_val)
    b2 = tf.Variable(b2_val)

    ############ Connect FC LAYERS ############################
    z1 = tf.matmul(input_FC1,W1) + b1
    h1 = tf.nn.relu(z1)

    ########## DROPOUT #########################
    prob = tf.placeholder(tf.float32)
    out_drop = tf.nn.dropout(h1,prob)

    z2 = tf.matmul(out_drop,W2) + b2
    h2 = tf.sigmoid(z2)

    sess  = tf.InteractiveSession()
    tf.initialize_all_variables().run()   ########### INTIALIZE VARIABLES ################

    val = []

    for j in xrange(0,len(testX),batchSize):   ######### TEST THE MODEL ########
        batchlen = min(batchSize,len(testX)-j)

        X = testX[j:j+batchlen]

        vals = sess.run(h2,feed_dict={x:X,prob:1.0})
        val.extend(vals)

    val = np.array(val)
    print "Testing of CNN Network completed\n"

    ans = np.array([1 if x>th else 0 for x in val]) 
    val = np.mean((ans == test_Y))*1.0

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



    with open("results/"+ resultsFile, "a") as text_file:
        print >> text_file, hyper
        print >> text_file, "True Positive, False Positive, False Negative, True Negative"
        print >> text_file, P1A1, P1A0, P0A1, P0A0
        print >> text_file, 'BucketSize : ', input_width
        print >> text_file ,'Accururacy : '+ str(val)
        print >> text_file ,'Precision : '+ str(sk.precision_score(test_Y,ans))
        print >> text_file, 'Recall : '+ str(sk.recall_score(test_Y,ans))
        print >> text_file, 'F1 : '+ str(sk.f1_score(test_Y,ans))
        print >>text_file, '\n'

    return ans   ################# RETURN PREDICTED LABELS ################




	