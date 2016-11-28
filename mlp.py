from hist import get_data

import numpy as np
import tensorflow as tf


t, v, test = get_data()

X_train, y_train = t
X_validation, y_validation = v
X_test, y_test = test

# help function to sampling data
def get_sample(num_samples, X_data, y_data):
	positions = np.arange(len(y_data))
	np.random.shuffle(positions)

	X_sample = []
	y_sample = []

	for posi in positions[:num_samples]:
		X_sample.append(X_data[posi])
		y_sample.append(y_data[posi])

	return X_sample, y_sample


######################## creating the model architecture #######################################

num_nodes_hidden_layer = 300

#NOTES:
# - Inicialmente: 19%
# - Mudando a camada escondida para sigmoid, conseguimos 21-22%
# - Mudando o numero de neuronios na camada escondida para 20 nos deu uma acuracia de 23% nos dados de validacao
# - Mudando para 10 neuronios, conseguimos 21%
# - Paramos de usar histogramas para alimentar as imagens em preto e branco
# - Com 300 neuronios, conseguimos 25% de acuracia
# - Com 100 neuronios, conseguimos 27%
# - Com 2 camadas de 200 neuronios, conseguimos 27%
# - Com 2 camadas de 100 neuronios, conseguimos 28%
# - Voltamos com 1 camada com 200 neuronios
# - Adicionamos regularizacao l2 0.001, conseguimos 30% (slides fred)
# - Mudamos a regularizacao para 0.01, conseguimos 32%
# - Mudamos a regularizacao para 0.005, conseguimos 32%
# - Mudamos para 2 camadas com 100 neuronios novamente
# - Colocamos regularizacao 0.0025, conseguimos 35%
# - Mudando a taxa de aprendizado para 0.1 e o tamanho do batch para 50, conseguimos 36%
# - Adicionamos outra camada de 100, conseguimos 25%
# - Tentando o algoritmo KNN nas imagens raw, conseguimos 16.5% NO TESTE
# - Tentando o algoritmo KNN nos histogramas, conseguimos 7% NO TESTE


# input placeholder
x = tf.placeholder(tf.float32, [None, 324])

# output placeholder
y_ = tf.placeholder(tf.float32, [None, 10])


# weights of the neurons in first layer
W1 = tf.Variable(tf.random_normal([324, 324], stddev=0.35))
b1 = tf.Variable(tf.random_normal([324], stddev=0.35))

# weights of the neurons in second layer
W2 = tf.Variable(tf.random_normal([324,162], stddev=0.35))
b2 = tf.Variable(tf.random_normal([162], stddev=0.35))

# weights of the neurons in third layer
W3 = tf.Variable(tf.random_normal([162,81], stddev=0.35))
b3 = tf.Variable(tf.random_normal([81], stddev=0.35))

# weights of the neurons in fourth layer
W4 = tf.Variable(tf.random_normal([81,10], stddev=0.35))
b4 = tf.Variable(tf.random_normal([10], stddev=0.35))


# hidden_layer value
#hidden_layer = tf.nn.softmax(tf.matmul(x, W1) + b1)
hidden_layer1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)  #Mudamos de softmax para sigmoid
hidden_layer2 = tf.nn.sigmoid(tf.matmul(hidden_layer1, W2) + b2)  #Mudamos de softmax para sigmoid
hidden_layer3 = tf.nn.sigmoid(tf.matmul(hidden_layer2, W3) + b3)

#sigmoid, sigmoid, tanh: 0.465
#sigmoid, tanh, sigmoid: 0.481
#tanh, sigmoid, sigmoid: 0.488
#tanh, tanh, sigmoid: 0.496
#tanh, sigmoid, tanh: 0.486
#sigmoid, tanh, tanh: 0.51
#tanh, tanh, tanh: 0.486
#sigmoid, sigmoid, sigmoid: 0.411

#usando tann conseguimos 26.9%
#hidden_layer1 = tf.nn.tanh(tf.matmul(x, W1) + b1)  #Tentando usar Tanh
#hidden_layer2 = tf.nn.tanh(tf.matmul(hidden_layer1, W2) + b2)  #Tentando usar Tanh

# output of the network
y_estimated = tf.nn.softmax(tf.matmul(hidden_layer3, W4) + b4)


# function to measure the error
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_estimated), reduction_indices=[1]))

cost = cross_entropy
cost += 0.0025*tf.reduce_sum(tf.square(W1))
cost += 0.0025*tf.reduce_sum(tf.square(W2))
cost += 0.0025*tf.reduce_sum(tf.square(W3))
cost += 0.0025*tf.reduce_sum(tf.square(W4))


# how to train the model
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


# how to evaluate the model
correct_prediction = tf.equal(tf.argmax(y_estimated,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


######################## training the model #######################################

# applying a value for each variable (in this case W and b)
init = tf.initialize_all_variables()


# a session is dependent of the enviroment where tensorflow is running
sess = tf.Session()
sess.run(init)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

#knn.fit(X_train, y_train)
#print "knn val score: {}".format(knn.score(X_validation, y_validation))
#print "knn test score: {}".format(knn.score(X_test, y_test))

num_batch_trainning = 50
total_rounds_of_trainning = 20000
for i in range(total_rounds_of_trainning): # trainning 1000 times

	# randomizing positions
	X_sample, y_sample = get_sample(num_batch_trainning, X_train, y_train)

	# where the magic happening
	sess.run(train_step, feed_dict={x: X_sample, y_:  y_sample})

	# print the accuracy result
	if i % 100 == 0:
		print i, "of", total_rounds_of_trainning, ": ",
                val = (sess.run(accuracy, feed_dict={x: X_validation, y_: y_validation}))	
                train =(sess.run(accuracy, feed_dict={x: X_train, y_: y_train})) 
                print "train={}, val={}".format(train, val)

print "\n\n\n"
print "TEST RESULT: ", (sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))
