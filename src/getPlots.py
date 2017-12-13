import numpy as np 
import matplotlib.pyplot as plt 
import Sentence_Similarity
import BucketedClassifierwithCNN as bcnn


def main():
	_,_,_,_,x,y = Sentence_Similarity.load_data_for_plotting()

	X = [np.array(Sentence_Similarity.getDynamicPooledMatrix(j)) for j  in x]
	Y = y

	bcnn.set_global_var(15,15)
	print np.shape(X)
	print np.shape(Y)
	y_pred = bcnn.test(X,Y)

	for i in range(0,len(Y)):
		plt.figure()
		plt.imshow(x[i],vmin=-1,vmax=1)
		plt.colorbar()


		if y[i] == 1:
			if y_pred[i] == 1:
				plt.savefig('results/images/test/A1P1/'+str(i))
			else:
				plt.savefig('results/images/test/A1P0/'+str(i))

		else:
			if y_pred[i] == 1:
				plt.savefig('results/images/test/A0P1/'+str(i))
			else:
				plt.savefig('results/images/test/A0P0/'+str(i))

		plt.close()



if __name__ == "__main__":
	main()