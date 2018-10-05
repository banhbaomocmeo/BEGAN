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
        x = norm_img(self.x)
        self.z = tf.placeholder(dtype=tf.float32, shape=[None, self.z_dim])
        g_img, self.g_vars = Generator(self.z, self.start_size, self.filters, self.blocks)
        d_img, self.d_vars, self.embbed = Discriminator(tf.concat([g_img, x], 0), self.z_dim, self.start_size, self.filters, self.blocks)
        AE_g, AE_x = tf.split(d_img, 2)

        self.g_img = denorm_img(g_img)
        self.AE_g = denorm_img(AE_g)
        self.AE_x = denorm_img(AE_x)
        
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_g - g_img))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_g - g_img))


        # self.g_opt = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=self.g_vars)
        # self.d_opt = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, var_list=self.d_vars)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        # with tf.control_dependencies([self.d_opt, self.g_opt]):
        #     self.k_update = tf.assign(
        #         self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        # self.summary_op = tf.summary.merge([
        #     tf.summary.image("G-AE_G-AE_x", tf.concat([self.g_img, self.AE_g, self.AE_x], 2)),

        #     tf.summary.scalar("loss/d_loss", self.d_loss),
        #     tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
        #     tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
        #     tf.summary.scalar("loss/g_loss", self.g_loss),
        #     tf.summary.scalar("misc/measure", self.measure),
        #     tf.summary.scalar("misc/k_t", self.k_t),
        #     tf.summary.scalar("misc/balance", self.balance),
        # ])



        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
    
    def test_model(self):
        self.sess.run(tf.global_variables_initializer())

