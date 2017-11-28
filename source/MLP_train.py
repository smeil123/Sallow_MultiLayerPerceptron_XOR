import random
import numpy as np
import pickle as pkl

print "xor multi layer perceptron"

# [0,0]	[1,0] [0,1] [1,1]
dataSet = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
correctSet = np.array([0,1,1,0])

# logistic activation
def activation(z):
    return (1.0 / (1.0 + np.exp(-z))).astype('float32')

def save(fn,obj):
    fd = open(fn,'wb')
    pkl.dump(obj,fd)
    fd.close() 

# multi layer perceptron funcion
def MSP():

	fd = open("train_log.txt",'w')

	weigth_11 = np.array([random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)]).astype('float32')
	weigth_12 = np.array([random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)]).astype('float32')

	# hidden layer weigth initialization
	weight_1 = np.array([weigth_11,weigth_12]).astype('float32')
	# hidden weigth => 3*2
	weight_1 = weight_1.reshape(3,2)

	# output layer weigth initialization
	# output weigth => 1*3
	weight_2 = np.array([random.uniform(-1.0,1.0),random.uniform(-1.0,1.0),random.uniform(-1.0,1.0)]).astype('float32')

	one_array = np.ones((4)).astype('float32')
	bias = one_array.reshape(4,1)

	for i in range(0,1000):
		error = 0;

		##### hidden layer 
		# (4*3,3*2) => 4*2
		hidden = np.dot(dataSet,weight_1)
		# g(hidden) + bias -> output layer input data
		hidden_out = np.concatenate((bias,activation(hidden)),axis=1)

		##### output layer
		# 4*3 3*1 => 4*1
		output = np.dot(hidden_out,weight_2)
		#result 
		y = activation(output)


		for j in range(0,4):
			# output >= 0.5 -> 1, output < 0.5 -> 0
			if(y[j] >= 0.5):
				if(correctSet[j] != 1):
					error = error + 1
			else:
				if(correctSet[j] != 0):
					error = error + 1

		print 'output',y
		print i,'th error number ->', error
		cost = 0.5 * np.dot((correctSet - y),(correctSet-y))
		fd.write('%d th cost -> %f\n' % (i,cost))
		fd.write('%d th error number ->%d\n' % (i,error))
		if error == 0:
			print i," th stop (error = 0) --------------------"
			fd.write('hidden layer parameter -> [[%f,%f,%f],[%f,%f,%f]]\n'% (weight_1[0,0],weight_1[1,0],weight_1[2,0],weight_1[0,1],weight_1[1,1],weight_1[2,1]))
			fd.write('output layer parameter -> [%f,%f,%f]\n'% (weight_2[0],weight_2[1],weight_2[2]))
			fd.close()
			save('best_param_1.pkl',weight_1)
			save('best_param_2.pkl',weight_2)
			break;	

		# (y-d)
		# 1*4
		delta_1 = np.multiply((correctSet-y),np.multiply(y,(one_array-y)))
		#### output layer weight update
		p_weight_2 = weight_2		
		weight_2 = weight_2 + np.dot(delta_1,hidden_out)

		#4*3
		delta_21 = np.multiply(delta_1.reshape(4,1),np.multiply((np.multiply(hidden_out,(np.ones((12)).reshape(4,3)-hidden_out))),p_weight_2))
		#### hidden layer weight update
		weight_1[:,0] = weight_1[:,0] + 2*np.dot(delta_21[:,1],dataSet)
		weight_1[:,1] = weight_1[:,1] + 2*np.dot(delta_21[:,2],dataSet)

	fd.close()
	return;


MSP()