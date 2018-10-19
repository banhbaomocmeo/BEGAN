from ultis import *
import tensorflow as tf
import numpy as np
import cv2
import os

class BEGAN():
    def __init__(self, image_size=64, z_dim=64, gamma=0.5, batch_size=32):

        self.batch_size = 32
        self.save_step = 1000
        self.lr_update_step = 75000
        self.gamma = gamma
        self.lr = tf.Variable(initial_value=0.00008, trainable=False, name='lr')
        self.lambda_k = 0.001
        self.k_t = tf.Variable(initial_value=0., trainable=False, name='anneal_factor')
        self.z_dim = z_dim
        self.filters = 64
        self.blocks = 3
        self.image_size = image_size
        self.start_size = self.image_size // 2**(self.blocks-1)
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, 3])
        # x = norm_img(self.x)
        self.z = tf.random_uniform(
                (tf.shape(self.x)[0], self.z_dim), minval=-1.0, maxval=1.0)        
        g_img, self.g_vars = Generator(self.z, self.start_size, self.filters, self.blocks)
        d_img, self.d_vars, self.embbed = Discriminator(tf.concat([g_img, self.x], 0), self.z_dim, self.start_size, self.filters, self.blocks)
        AE_g, AE_x = tf.split(d_img, 2)

        self.g_img = denorm_img(g_img)
        self.AE_g = denorm_img(AE_g)
        self.AE_x = denorm_img(AE_x)
        
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - self.x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_g - g_img))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_g - g_img))


        self.g_opt = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=self.g_vars)
        self.d_opt = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, var_list=self.d_vars)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([self.d_opt, self.g_opt]):
            k_clip = tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1)
            self.k_update = tf.assign(self.k_t, k_clip)
        self.lr_update = tf.assign(self.lr, tf.maximum(1E-6, self.lr / 2))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G-AE_G-AE_x", tf.concat([self.g_img, self.AE_g, self.AE_x], 2)),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/balance", self.balance),
        ])


        
    def build_interpolated_model(self):
        with tf.variable_scope("interpalation") as vs:
            # Extra ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_dim], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _ = Generator(self.z_r, self.start_size, self.filters, self.blocks, reuse=True)

        with tf.variable_scope("interpalation_opt") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

    def init_var(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, iters=200000):
        n_batch = X.shape[0] // self.batch_size
        id_batch = 0
        batch = None
        np.random.shuffle(X)
        self.sess.run(tf.global_variables_initializer())
        for i in range(iters):
            if id_batch == n_batch:
                batch = X[id_batch*self.batch_size:]
                np.random.shuffle(X)
                id_batch = 0
            else:
                batch = X[id_batch*self.batch_size: (id_batch+1)*self.batch_size]
                id_batch += 1
            z = np.random.uniform(-1, 1, size=(batch.shape[0], self.z_dim))

            _, _, d_loss, g_loss, k_t, measure, balance = self.sess.run([self.g_opt, self.d_opt, self.d_loss, self.g_loss, self.k_update, self.measure, self.balance],
                            feed_dict={self.x: batch, self.z: z}
                        )
            print('iter {} - d_loss: {}, g_loss: {}, measure: {}, balance: {}, k_t: {}'.format(i, d_loss, g_loss, measure, balance, k_t))
            if i % self.save_step == self.save_step - 1: 
                self.saver.save(self.sess, "./model/model.ckpt")               
                summ, g_img, AE_g, AE_x = self.sess.run([self.summary_op, self.g_img, self.AE_g, self.AE_x],
                            feed_dict={self.x: batch, self.z: z}
                        )
                self.summary_writer.add_summary(summ, i // self.save_step)
                img = np.concatenate((g_img, AE_g, AE_x), axis=1)
                cv2.imwrite('./images/iter_{}.jpg'.format(i), np.hstack(img)[:,:,::-1])
            if i % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.lr_update)
            
    def train_interpolation(self, batch1, batch2, step=0, train_epoch=500, root_path='./interp'):
        batch_size = len(batch1)
        self.sess.run(self.z_r_update)
        for i in range(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: np.vstack([batch1, batch2])})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:batch_size], z[batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.sess.run(self.g_img, feed_dict={self.z: z})
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)



                