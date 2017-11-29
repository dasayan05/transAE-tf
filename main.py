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

    capl = CapsuleLayer(30)
    y_ = capl(x, dxy)
    
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(y-y_))
        # loss_summary = tf.summary.scalar('loss', loss)

    with tf.name_scope('optim'):
        optimizer = tf.train.AdamOptimizer()
        train_step = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("transAE_Log")
        writer.add_graph(graph=sess.graph)

        bsize = 128
        Epochs = 100

        saver = tf.train.Saver(max_to_keep=2)

        # gstep = 0
        for E in range(Epochs):
            avgloss = 0.

            print('epoch {0} starts'.format(E))
            for I in range(int(55000/bsize)):
                X, _ = mnist.train.next_batch(bsize, shuffle=True)
                Xs, R = shift_batch_rand(X)
                l, _ = sess.run([loss, train_step], feed_dict={x: X, dxy: R, y: Xs})
                avgloss = ((avgloss*I) + l)/(I + 1)
                if I % 100 == 0:
                    print('epoch {0} iteration {1}'.format(E, I))
                    # writer.add_summary(lsum, gstep)
                # gstep += 1

            print('epoch {0} avgloss {1}'.format(E, avgloss))
            saver.save(sess, 'transaemodels/model{0}'.format(E))


if __name__ == '__main__':
    if os.path.exists("transAE_Log"):
        shutil.rmtree("transAE_Log")
    main()