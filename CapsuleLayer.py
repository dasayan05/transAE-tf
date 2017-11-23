from Capsule import Capsule
import tensorflow as tf

class CapsuleLayer( object ):
    'a layer of multiple capsules'

    def __init__(self, num_caps, input_dim = 784, capsule_dim = 10, gen_dim = 20, dtype=tf.float64):
        'defines a CapsuleLayer of multiple capsules'
        # num_caps: number of capsules
        # input_dim: input vector dimension (default to MNIST)
        # capsule_dim: dimension of a capsule (recognition units)
        # gen_dim: generation dimension (generation units)

        # first things first - track inputs
        self.num_caps = num_caps

        self.input_dim = input_dim
        self.capsule_dim = capsule_dim
        self.gen_dim = gen_dim

        # construct the capsules
        self.caps = []
        for _ in range(self.num_caps):
            self.caps.append( Capsule(self.input_dim, self.capsule_dim, self.gen_dim, dtype) )

    def __call__(self, x: 'input tensor', dxy: 'shift tensor'):
        'The forward propagation'
        # input: input tensor (batch x input_dim)
        # delta_xy: deltaX and deltaY as (batch x 2)
        # separate_prob: if True, returns the prob separately

        # get outputs of all the capsules
        cap_outs = [one_cap(x, dxy) for one_cap in self.caps]

        with tf.name_scope('add_caps'):
            # the final reconstruction of the shifted image
            out = tf.add_n(cap_outs, name='final_out')

        return out