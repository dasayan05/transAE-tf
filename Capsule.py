import tensorflow as tf

class Capsule( object ):
    """ The capsules described in the paper 'Tranforming Autoencoders,
    G.E. Hinton et al' http://www.cs.toronto.edu/~fritz/absps/transauto6.pdf """

    __n_caps = 0

    def __init__(self, input_dim = 784, capsule_dim = 10, gen_dim = 20, dtype=tf.float64):
        'Defines a capsule object that models \
        the lower-level patches and their translation'
        # input_dim: input vector dimension (default to MNIST)
        # capsule_dim: dimension of a capsule (recognition units)
        # gen_dim: generation dimension (generation units)

        # increase global object count
        Capsule.__n_caps += 1
        self.name_prefix = 'cap' + str(Capsule.__n_caps)
        
        # first things first - track inputs
        self.input_dim = input_dim
        self.capsule_dim = capsule_dim
        self.gen_dim = gen_dim
        self.xy_dim = 2 # we have two scalars - x, y

        # create the weights
        with tf.variable_scope(self.name_prefix + 'var'):
            self.__init_weights__(dtype)

    def __call__(self, input, delta_xy, separate_prob = False):
        'The forward propagation'
        # input: input tensor (batch x input_dim)
        # delta_xy: deltaX and deltaY as (batch x 2)
        # separate_prob: if True, returns the prob separately
        # returns:
        #      reco: reconstruction tensor
        #      prob: probability of having the entity of this capsule

        with tf.name_scope(self.name_prefix):
            self.cap = self.__apply_layer('cap', input, self.w_cap, self.b_cap, activation=tf.nn.sigmoid)
            self.xy = self.__apply_layer('xy', self.cap, self.w_xy, self.b_xy)
            self.pr = self.__apply_layer('pr', self.cap, self.w_pr, self.b_pr, activation=tf.nn.sigmoid)
            self.gen = self.__apply_layer('gen', self.xy + delta_xy, self.w_gen, self.b_gen, activation=tf.nn.sigmoid)
            self.reco = self.__apply_layer('reco', self.gen, self.w_rec)

            if separate_prob:
                return self.reco, self.pr
            else:
                return tf.multiply(self.reco, self.pr)

    def __apply_layer(self, name, input, weight, bias = None, activation = tf.identity):
        'get the output tensors from inputs'

        if bias != None:
            return activation( tf.matmul(input, weight) + bias, name=self.name_prefix+name )
        else:
            return activation( tf.matmul(input, weight), name=self.name_prefix+name )

    def __init_weights__(self, dtype):
        'creates all the required weight variables'
        # dtype: type of the weights

        # params of recognition unit
        self.w_cap, self.b_cap = self.__create_w_b__(self.input_dim, self.capsule_dim, 'cap')

        # params of coordinate unit
        self.w_xy, self.b_xy = self.__create_w_b__(self.capsule_dim, self.xy_dim, 'xy')

        # params of probability unit
        self.w_pr, self.b_pr = self.__create_w_b__(self.capsule_dim, 1, 'pr')

        # params of generation unit
        self.w_gen, self.b_gen = self.__create_w_b__(self.xy_dim, self.gen_dim, 'gen')

        # params of the reconstruction unit
        self.w_rec = self.__create_w_b__(self.gen_dim, self.input_dim, 'rec', have_bias=False)

    def __create_w_b__(self, dim_0, dim_1, postfix: 'str', have_bias=True, dtype=tf.float64):
        'helper function for easy making weight-bias pair'

        w = tf.get_variable('w_' + postfix, shape=(dim_0, dim_1), dtype=dtype,
            initializer=tf.initializers.random_normal)

        if have_bias: # if want bias
            b = tf.get_variable('b_' + postfix, shape=(dim_1,), dtype=dtype,
                initializer=tf.initializers.zeros)
            return w, b

        return w