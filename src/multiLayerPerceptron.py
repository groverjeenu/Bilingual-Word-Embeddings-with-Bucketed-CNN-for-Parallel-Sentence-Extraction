################################### CLASSIFIER ##################
import numpy as np
import sklearn.metrics as sk
import tensorflow as tf

no_of_hidden_units = 100
no_of_epochs = 100
batchSize = 1
eta = 1

hyper = 'simple_dynamic_pool_dense_' + str(no_of_hidden_units) + '_' + str(no_of_epochs) + '_' + str(
    batchSize) + '_' + str(eta)


def train(trainX, trainY):
    perm = np.random.permutation(trainY.shape[0])
    trainX = trainX[perm]
    trainY = trainY[perm]

    ################# NETWORK DIMENSIONS #####################
    no_of_inputs = np.shape(trainX)[0]
    print np.shape(trainX)
    input_size = len(np.ndarray.flatten(np.array(trainX[0], copy=True)))
    print input_size
    output_size = 1

    trainY = np.array([np.array([x]) for x in trainY])
    print np.shape(trainY)

    ############### MODEL ##################

    W1 = tf.Variable(tf.random_normal([input_size, no_of_hidden_units], -1), name='W1') * (
            1 / np.sqrt(no_of_hidden_units * input_size))
    b1 = tf.Variable(tf.zeros([no_of_hidden_units]), name='b1')

    W2 = tf.Variable(tf.random_normal([no_of_hidden_units, output_size], -1), name='W2') * (
            1 / np.sqrt(output_size * no_of_hidden_units))
    b2 = tf.Variable(tf.zeros([output_size]), name='b2')

    x = tf.placeholder(tf.float32, [None, input_size])
    y = tf.placeholder(tf.float32, [None, output_size])

    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.sigmoid(z1)

    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.sigmoid(z2)

    loss = tf.reduce_mean(tf.square(y - a2))

    train_step = tf.train.GradientDescentOptimizer(eta).minimize(loss)

    sess = tf.InteractiveSession()

    tf.initialize_all_variables().run()

    for e in range(0, no_of_epochs):

        for j in xrange(0, no_of_inputs, batchSize):
            batchlen = min(batchSize, no_of_inputs - j)

            X = trainX[j:j + batchlen]
            Y = trainY[j:j + batchlen]

            sess.run(train_step, feed_dict={x: X, y: Y})
            print (sess.run([loss], feed_dict={x: X, y: Y}))

    np.save('weights/w1' + hyper, W1.eval())
    np.save('weights/w2' + hyper, W2.eval())
    np.save('weights/b1' + hyper, b1.eval())
    np.save('weights/b2' + hyper, b2.eval())


def test(testX, test_Y, th=0.5):
    testX = [np.ndarray.flatten(x) for x in testX]
    input_size = len(np.ndarray.flatten(np.array(testX[0], copy=True)))

    w1 = np.load('weights/w1' + hyper + '.npy')
    bb1 = np.load('weights/b1' + hyper + '.npy')
    w2 = np.load('weights/w2' + hyper + '.npy')
    bb2 = np.load('weights/b2' + hyper + '.npy')

    W1 = tf.constant(w1)
    b1 = tf.constant(bb1)

    W2 = tf.constant(w2)
    b2 = tf.constant(bb2)
    x = tf.placeholder(tf.float32, [None, input_size])

    z1 = tf.matmul(x, W1) + b1
    a1 = tf.nn.sigmoid(z1)
    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.nn.sigmoid(z2)

    sess = tf.InteractiveSession()
    tf.initialize_all_variables().run()

    val = []
    for i in xrange(0, len(testX), batchSize):
        batchlen = min(batchSize, len(testX) - i)
        vals = sess.run(a2, feed_dict={x: testX[i:i + batchlen]})
        val.extend(vals)

    ans = np.array([1 if x > th else 0 for x in val])
    val = np.mean((ans == test_Y)) * 1.0

    print 'Accuracy : ', val
    print 'Recall : ', sk.recall_score(test_Y, ans)
    print 'Precision : ', sk.precision_score(test_Y, ans)

    cnt = 0
    cntT = 0
    for i in range(0, len(test_Y)):
        if ans[i] == 1:
            cnt += 1
            if test_Y[i] == 1:
                cntT += 1

    P1A1 = P0A0 = P1A0 = P0A1 = 0
    for i in range(0, len(test_Y)):
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

    with open("results/" + 'classsifier_MLP_results', "a") as text_file:
        print >> text_file, hyper
        print >> text_file, "True Positive, False Positive, False Negative, True Negative"
        print >> text_file, P1A1, P1A0, P0A1, P0A0
        print >> text_file, 'Accururacy : ' + str(val)
        print >> text_file, 'Precision : ' + str(sk.precision_score(test_Y, ans))
        print >> text_file, 'Recall : ' + str(sk.recall_score(test_Y, ans))
        print >> text_file, 'F1 : ' + str(sk.f1_score(test_Y, ans))
        print >> text_file, '\n'

    return ans
