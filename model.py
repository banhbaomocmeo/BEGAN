from ultis import *
import tensorflow as tf

class BEGAN():
    def __init__(self):
        self.gamma = 0.5
        self.lr = 0.00008
        self.lambda_k = 0.001
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
        AE_g, AE_x = tf.split(d_img, 2)
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - self.x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_g - g_img))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_g - g_img))


        self.g_opt = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=self.g_vars)
        self.d_opt = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, var_list=self.d_vars)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([self.d_opt, self.g_opt]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))