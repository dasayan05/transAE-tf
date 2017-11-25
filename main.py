from CapsuleLayer import CapsuleLayer
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from numpy.random import randn
from utils import shift_batch_rand
import os, shutil

def main( ):
    # dataset
    mnist = input_data.read_data_sets("../MNIST_data/")

    x = tf.placeholder(tf.float64, shape=(None,784), name='x')
    dxy = tf.placeholder(tf.float64, shape=(None,2), name='dxy')
    y = tf.placeholder(tf.float64, shape=(None,784), name='xpdxy')

    capl = CapsuleLayer(5)
    y_ = capl(x, dxy)
    
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y-y_))

    with tf.name_scope('optim'):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("transAE_Log")
        writer.add_graph(graph=sess.graph)

        for _ in range(10):
            X, _ = mnist.train.next_batch(100, shuffle=True)
            Xs, R = shift_batch_rand(X)
            q = sess.run(loss, feed_dict={x: X, dxy: R, y: Xs})
            print(q)

if __name__ == '__main__':
    if os.path.exists("transAE_Log"):
        shutil.rmtree("transAE_Log")
    main()