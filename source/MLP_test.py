import random
import pickle as pkl
import numpy as np

# [0,0]	[1,0] [0,1] [1,1]
dataSet = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
correctSet = np.array([0,1,1,0])
bias = np.ones((4)).reshape(4,1)
# logistic activation
def activation(z):
    return (1.0 / (1.0 + np.exp(-z))).astype('float32')
	
def load(fn):
    fd = open(fn,'rb')
    obj = pkl.load(fd)
    fd.close()
    return obj

#### test xor
weight_1 = load('best_param_1.pkl')
weight_2 = load('best_param_2.pkl')
h1 = activation(np.dot(dataSet,weight_1))
hh1 = np.concatenate((bias,h1),axis=1)
h2 = activation(np.dot(hh1,weight_2))
error = 0

fd = open("test_output.txt",'w')

print "XOR Test->"
for j in range(0,4):
	if(h2[j] >= 0.5):
		temp = 1
	else:
		temp = 0
	if(correctSet[j] != temp):
		fd.write('y --> %d correct --> %d ==> false\n' % (temp,correctSet[j]))
		print "y-->",temp,"correct -->",correctSet[j],"==> false"
	else:
		fd.write('y --> %d correct --> %d ==> correct!!\n'% (temp,correctSet[j]))
		print "y-->",temp,"correct -->",correctSet[j],"==> correct"