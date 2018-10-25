from ultis import *
import tensorflow as tf
import numpy as np
import cv2
import os
from interpolations import create_mine_grid

class BEGAN():
    def __init__(self, image_size=64, z_dim=64, gamma=0.5, batch_size=32, num_classes=500, delta=1):

        self.batch_size = 32
        self.save_step = 1000
        self.lr_update_step = 75000
        self.gamma = gamma
        self.delta = delta
        self.lr = tf.Variable(initial_value=0.00008, trainable=False, name='lr')
        self.lambda_k = 0.001
        self.k_t = tf.Variable(initial_value=0., trainable=False, name='anneal_factor')
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.filters = 64
        self.blocks = 3
        self.image_size = image_size
        self.start_size = self.image_size // 2**(self.blocks-1)

    def build_model(self):
        
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.image_size, self.image_size, 3], name='real_inputs')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name='real_labels')
        # x = norm_img(self.x)
        self.training = tf.placeholder(dtype=tf.bool, shape=None, name='training_flag')
        self.z = tf.random_uniform(
                (tf.shape(self.x)[0], self.z_dim), minval=-1.0, maxval=1.0)    
        self.yz = tf.concat([self.y, self.z], axis=1)    
        g_img, self.g_vars = Generator(self.yz, self.start_size, self.filters, self.blocks)
        d_img, self.d_vars, self.fl = Discriminator(tf.concat([g_img, self.x], 0), self.z_dim, self.start_size, self.filters, self.blocks)
        AE_g, AE_x = tf.split(d_img, 2)

        #classifier
        q_out, self.q_vars = Classifier(self.fl, self.num_classes, training=self.training)
        q_g, q_x = tf.split(q_out, 2)
        self.q_loss_real = tf.reduce_mean(smce_lg(q_x, self.y))
        self.q_loss_fake = tf.reduce_mean(smce_lg(q_g, self.y))

        self.g_img = denorm_img(g_img)
        self.AE_g = denorm_img(AE_g)
        self.AE_x = denorm_img(AE_x)
        
        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - self.x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_g - g_img))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake + self.delta * (self.q_loss_real + self.q_loss_fake)
        self.g_loss = self.d_loss_fake + self.delta * self.q_loss_fake


        self.g_opt = tf.train.AdamOptimizer(self.lr).minimize(self.g_loss, var_list=self.g_vars)
        self.d_opt = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss, var_list=[self.d_vars, self.q_vars])

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
            tf.summary.scalar("loss/q_loss_real", self.q_loss_real),
            tf.summary.scalar("loss/q_loss_fake", self.q_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/balance", self.balance),
        ])
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                    gpu_options=gpu_options)
        self.sess = tf.Session(config=sess_config)
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter('logs', self.sess.graph)
        
    def init_var(self):
        self.sess.run(tf.global_variables_initializer())

    def load_model(self):
        self.saver.restore(self.sess, 'model/model.ckpt')

    def fit(self, X, Y, iters=200000):
        n_batch = X.shape[0] // self.batch_size
        id_batch = 0
        batch = None
        X, Y = shuffle(X, Y)
        self.sess.run(tf.global_variables_initializer())
        for i in range(iters):
            if id_batch == n_batch:
                batch = (X[id_batch*self.batch_size:], Y[id_batch*self.batch_size:])
                X, Y = shuffle(X, Y)
                id_batch = 0
            else:
                batch = (X[id_batch*self.batch_size: (id_batch+1)*self.batch_size], Y[id_batch*self.batch_size: (id_batch+1)*self.batch_size])
                id_batch += 1
            z = np.random.uniform(-1, 1, size=(batch[0].shape[0], self.z_dim))

            _, _, d_loss, g_loss, k_t, measure, balance = self.sess.run([self.g_opt, self.d_opt, self.d_loss, self.g_loss, self.k_update, self.measure, self.balance],
                            feed_dict={self.x: batch[0], self.y: batch[1], self.z: z, self.training: True}
                        )
            print('iter {} - d_loss: {}, g_loss: {}, measure: {}, balance: {}, k_t: {}'.format(i, d_loss, g_loss, measure, balance, k_t))
            if i % self.save_step == self.save_step - 1: 
                self.saver.save(self.sess, "./model/model.ckpt", global_step=i)               
                summ, g_img, AE_g, AE_x = self.sess.run([self.summary_op, self.g_img, self.AE_g, self.AE_x],
                            feed_dict={self.x: batch[0], self.y: batch[1], self.z: z, self.training: False}
                        )
                self.summary_writer.add_summary(summ, i // self.save_step)
                img = np.concatenate((g_img, AE_g, AE_x), axis=1)
                cv2.imwrite('./images/iter_{}.jpg'.format(i), np.hstack(img)[:,:,::-1])
            if i % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run(self.lr_update)

    def generate(self, X, Y, size=5):
        batch_size = X.shape[0]
        zs = create_mine_grid(rows=size, cols=size, dim=self.z_dim, space=1, anchors=None, spherical=True, gaussian=True)
        outputs = []
        for z in zs:
            yz = np.concatenate([Y, batch_size*[z]], axis=1)
            output = self.sess.run(self.g_img, feed_dict={self.yz: yz})
            # print('>>>>>>output', output.shape)
            outputs.append(output)
        outputs = np.array(outputs)
        # print('>>>>>>outputs', outputs.shape)
        outputs[0] = (X+1)*127.5 
        for i in range(batch_size):
            img = outputs[:, i]
            # print('>>>>>>img', img.shape)
            make_image_from_batch(img, './generate/img_{}.jpg'.format(i))
            