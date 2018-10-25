import tensorflow as tf 
import numpy as np
from PIL import Image
import math



def conv2d(x, filters, kernel_size=3, strides=1, name=None):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding='same', name=name)

def bnorm(x, training=False, name=None):
    return tf.layers.batch_normalization(x, training=training, name=name)

def dropout(x, drate=0.5, training=False, name=None):
    return tf.layers.dropout(x, rate=drate, noise_shape=None, seed=None, training=training, name=name)

def elu(x, name=None):
    return tf.nn.elu(x, name=name)

def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)

def dense(x, units, name=None):
    return tf.layers.dense(x, units, name=name)

def smce_lg(x, y, name=None):
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=x, labels=y, name=name)


def get_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size, name=None):
    x = tf.image.resize_nearest_neighbor(x, new_size, name=name)
    return x

def upscale(x, scale, name=None):
    _, h, w, _ = get_shape(x)
    return resize_nearest_neighbor(x, (h*scale, w*scale), name=name)

def norm_img(image):
    image = image/127.5 - 1.
    return image

def denorm_img(norm):
    return tf.clip_by_value((norm + 1)*127.5, 0, 255)
    
def encoder(x, z_dim=64, filters=64, blocks=3, name='Encoder', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        for i in range(blocks):
            n = filters * (i+1)
            x = conv2d(x, n, 3, 1)
            x = elu(x)
            x = conv2d(x, n, 3, 1)
            x = elu(x)
            if(i < blocks - 1):
                x = conv2d(x, n, 3, 2)
                x = elu(x)
        fl = tf.layers.flatten(x)
        x = dense(fl, z_dim)

    return x, fl

def decoder(z, start_size=8 ,filters=64, blocks=3, name='Decoder', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = dense(z, start_size*start_size*filters)
        x = tf.reshape(x, [-1, start_size, start_size, filters])

        for i in range(blocks):
            x = conv2d(x, filters, 3, 1)
            x = elu(x)
            x = conv2d(x, filters, 3, 1)
            x = elu(x)
            if(i < blocks - 1):
                x = upscale(x, 2)

        x = conv2d(x, 3, 3, 1)
    return x

def Discriminator(x, z_dim=64, start_size=8, filters=64, blocks=3, name='Discriminator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        z, fl = x, fl = encoder(x, z_dim, reuse=reuse)
        x = decoder(z, start_size, reuse=reuse)
    var_list  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return x, var_list, fl

def Generator(z, start_size=8, filters=64, blocks=3, name='Generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = decoder(z, start_size, reuse=reuse)
    var_list  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return x, var_list 

def Classifier(x, num_class, training=False, name='Classifier', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = dense(x, 2048)
        x = bnorm(x, training)
        x = elu(x)
        x = dense(x, num_class)
    var_list  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return x, var_list
        


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0-val) * low + val * high # L'Hopital's rule/LERP
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega) / so * high


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

def shuffle(x, y):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    return x[index], y[index]
    
def make_image_from_batch(X, filename):
    '''
    this is document
    '''
    batch_size, h, w, c = X.shape
    no_col = int(np.ceil(np.sqrt(batch_size)))
    no_row = int(np.ceil(batch_size/no_col))
    output = np.zeros((int(no_row*h), int(no_col*w), c))
    for row in range(no_row):
        for col in range(no_col):
            if (row*no_col + col) == batch_size:
                break
            output[row*h:(row+1)*h,col*w:(col+1)*w] = X[row*no_col + col]
            
        if (row*no_col + col) == batch_size:
                break
    ndarr = np.squeeze(output).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(filename)