import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import os

def main( ):

    model_folder = os.path.join('.', 'transaemodels')
    saved_model = tf.train.latest_checkpoint(model_folder)
    tf.logging.debug('loading model from {0}'.format(saved_model))

    tf.reset_default_graph()
    with tf.Session(graph=tf.get_default_graph()) as sess:
        graph = tf.get_default_graph()

        # recover the graph and the parameters
        saver = tf.train.import_meta_graph(saved_model + '.meta')
        tf.logging.info('recovered the graph')
        saver.restore(sess, saved_model)
        tf.logging.info('recovered the parameters')

        # get the placeholders
        x = graph.get_tensor_by_name('x' + ':0')
        dxy = graph.get_tensor_by_name('dxy' + ':0')
        tf.logging.info('recovered the placeholders')

if __name__ == '__main__':
    main()