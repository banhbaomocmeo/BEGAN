import tensorflow as tf 

def conv2d(x, filters, kernel_size=3, strides=1, name=None):
    return tf.layers.conv2d(x, filters, kernel_size, strides, padding='same', name=name)

def elu(x, name=None):
    return tf.nn.elu(x, name=name)

def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)

def dense(x, units, name=None):
    return tf.layers.dense(x, units, name=name)

def get_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]

def resize_nearest_neighbor(x, new_size, name=None):
    x = tf.image.resize_nearest_neighbor(x, new_size, name=name)
    return x

def upscale(x, scale, name=None):
    _, h, w, _ = get_shape(x)
    return resize_nearest_neighbor(x, (h*scale, w*scale), name=name)

    
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
        x = tf.layers.flatten(x)
        x = dense(x, z_dim)

    return x

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
        z = x = encoder(x, z_dim, reuse=reuse)
        x = decoder(z, start_size, reuse=reuse)
    var_list  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return x, var_list, z

def Generator(z, start_size=8, filters=64, blocks=3, name='Generator', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        x = decoder(z, start_size, reuse=reuse)
    var_list  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return x, var_list 

