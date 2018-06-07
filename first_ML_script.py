# First Machine Learning script aiming at testing tensorflow module with MNIST example
# Author: Paul

#### Libraries

# Standard library
import tensorflow as tf
import mnist_loader
import numpy as np
import matplotlib.pyplot as plt

#Importing the data in different numpy.arrays
test,training_set=mnist_loader.load_data()


#Parameters
learning_rate=0.04
batch_size=20
epochs_nbr=10

###Definition of the graph
#Graph input

x=tf.placeholder(tf.float64, [None, 784])
y_true=tf.placeholder(tf.float64, [None, 10])

hidden_layer1=tf.layers.dense(x, units=100, activation=tf.nn.sigmoid)
hidden_layer2=tf.layers.dense(hidden_layer1, units=30, activation=tf.nn.sigmoid)
y_pred=tf.layers.dense(hidden_layer2,units=10,activation=tf.nn.softmax)

#Initialization of weights and bias
init=tf.global_variables_initializer()

#Definition of the loss
loss=tf.losses.mean_squared_error(labels=y_true,predictions=y_pred)

optimizer=tf.train.GradientDescentOptimizer(learning_rate)
train=optimizer.minimize(loss)


correct_prediction= tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
accuracy= tf.reduce_mean(tf.cast(correct_prediction,tf.float64))

#Start training

sess=tf.Session()
sess.run(init)

losses=np.zeros((epochs_nbr))
accuracies=np.zeros((epochs_nbr))

for epoch in range(epochs_nbr):
    #shuffle the training dataset
    np.random.shuffle(training_set)
    batchs_number=int(training_set.shape[0]/batch_size)
    
    for b in range(batchs_number):
        batch_x, batch_y= training_set[b*batch_size:(b+1)*batch_size,:784],\
        training_set[b*batch_size:(b+1)*batch_size,784:795]
        _,loss_batch= sess.run([train, loss], feed_dict={x: batch_x, y_true: batch_y})
        
        losses[epoch]+= loss_batch/ batchs_number
        
    print("Epoch n° {}".format(epoch), "loss={}".format(losses[epoch]))
    
    # Test of accuracy on data test
    accuracies[epoch]=accuracy.eval({x: test[:3000,:784], y_true: test[:3000,784:795]}\
              ,session=sess)
    
plt.figure(1)
plt.subplot(211)
plt.plot([i for i in range(epochs_nbr)], losses)

plt.subplot(212)
plt.plot([i for i in range(epochs_nbr)], accuracies)
plt.show()


    
        
    







