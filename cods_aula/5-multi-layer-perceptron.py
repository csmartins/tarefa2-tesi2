from hist_feature_test import *

import numpy as np
import tensorflow as tf


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


# input placeholder
#[None,10] > None pq nao sei quantas imagens terei. 768 pq as imagens sao 32x32 apos extrair o histograma de tamanho 768.
x = tf.placeholder(tf.float32, [None, 768])

# output placeholder
#saida esperada
#[None,10] > None pq nao sei quantas imagens terei. 10 eh pq tenho um vetor de 10 posicoes (10 classes), onde cada #imagem tera 1 na posicao de sua classe e zero no resto.
y_ = tf.placeholder(tf.float32, [None, 10])


# weights of the neurons in first layer
#Cada camada da rede tera um conjunto de pesos para que cada neuronio saiba fazer a combinacao linear de suas entradas.
# as redes vao melhorando os pesos iterativamente ate alcancar o esperado
#tf.random_normal -> distribuicao normal | [768, 300] -> formato da matriz do tensor | stddev=0.35 -> desvio padrao da distribuicao normal
W1 = tf.Variable(tf.random_normal([768, 300], stddev=0.35))
#faz parte da combinacao linear somar um termo extra 'bias', tem um por neuronio apenas
b1 = tf.Variable(tf.random_normal([300], stddev=0.35))

# weights of the neurons in second layer
W2 = tf.Variable(tf.random_normal([300,10], stddev=0.35))
b2 = tf.Variable(tf.random_normal([10], stddev=0.35))


# hidden_layer value
#tf.matmul(x, W1) multiplica matrizes. x eh a entrada e w1 eh a matriz de pesos.
#tf.nn.softmax funcao de ativacao da camada pq as camadas nao podem ter saida linear, aplica nao linearidade
#TENTAR USAR SIGMOIDE AO INVES DE SOFTMAX
#SOFTMAX faz todos os elementos somarem um e podemos perder informacao. Cria dependencia entre as saidas dos neuronios.
hidden_layer = tf.nn.softmax(tf.matmul(x, W1) + b1) 


# output of the network
#DEIXA O SOFTMAX: saida eh um vetor de tamanho 10. Softmax garante que a soma de todos os elementos eh sempre 1.
y_estimated = tf.nn.softmax(tf.matmul(hidden_layer, W2) + b2)


# function to measure the error
#diferenca entre o estimado e o esperado
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_estimated), reduction_indices=[1]))


# how to train the model
#PODE TENTAR MUDAR O TAMANHO DO PASSO (LEARNING RATE) 0.5 PRA MELHORAR
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# how to evaluate the model
correct_prediction = tf.equal(tf.argmax(y_estimated,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


######################## training the model #######################################

# applying a value for each variable (in this case W and b)
init = tf.initialize_all_variables()


# a session is dependent of the enviroment where tensorflow is running
sess = tf.Session()
sess.run(init)



num_batch_trainning = 500
for i in range(10000): # trainning 10000 times

	# randomizing positions
	X_sample, y_sample = get_sample(num_batch_trainning, X_train, y_train)

	# where the magic happening
	#seta os placeholders
	sess.run(train_step, feed_dict={x: X_sample, y_:  y_sample})

	# print the accuracy result
	if i % 100 == 0:
		print i, ": ", (sess.run(accuracy, feed_dict={x: X_validation, y_: y_validation}))	
	

print "\n\n\n"
print "TEST RESULT: ", (sess.run(accuracy, feed_dict={x: X_test, y_: y_test}))

