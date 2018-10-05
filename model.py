from ultis import *
import tensorflow as tf

class BEGAN():
    def __init__(self):
        self.gamma = 0.5
        self.lr = 0.00001
        self.k_t = 0
        self.z_dim = 64
        self.filters = 64
        self.blocks = 3
        self.image_size = 32
        self.start_size = self.image_size // 2**(self.blocks-1)
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, 3])
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        g_img, self.g_vars = Generator(self.z, self.start_size, self.filters, self.blocks)
        d_img, self.d_vars, self.embbed = Discriminator(tf.concat([g_img, self.x], 0), self.z_dim, self.start_size, self.filters, self.blocks)
        AE_G, AE_x = tf.split(d_img, 2)
        g_opt, d_opt = tf.train.AdamOptimizer(self.lr), tf.train.AdamOptimizer(self.lr)
