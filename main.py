from src import generateAndLoadMatrices
from src import multiLayerPerceptron
from src import config
from src import CNNBasedClassifier


def main():
    # Baseline : Multilayered Perceptron
    ####################################
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = generateAndLoadMatrices.load_Word_Vecs_for_Data()
    ######## Uncomment to retrain ###############
    # multiLayerPerceptron.train(train_X,train_Y)
    multiLayerPerceptron.test(test_X, test_Y)

    # CNN Classifier on Bucketed Data
    ##################################
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = generateAndLoadMatrices.load_Word_Vecs_for_Data_Bucket(
        config.buckets)

    for i in config.valid_buckets:
        ind = i
        dim = config.buckets[ind]
        CNNBasedClassifier.set_global_var(ind, dim, bucketed=True)
        ######## Uncomment to retrain ###############
        # CNNBasedClassifier.train(train_X[ind],train_Y[ind])
        CNNBasedClassifier.test(test_X[ind], test_Y[ind])

    # CNN Classifier on Non-bucketed Data
    #####################################
    ind = 15
    dim = 15
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = generateAndLoadMatrices.load_Word_Vecs_for_Data()
    CNNBasedClassifier.set_global_var(ind, dim)
    ######## Uncomment to retrain ###############
    # CNNBasedClassifier.train(train_X,train_Y)
    CNNBasedClassifier.test(test_X, test_Y)


if __name__ == "__main__":
    main()
