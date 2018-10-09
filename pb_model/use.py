import tensorflow as tf
# from common import one_hot, make_image_from_batch
import numpy as np
import matplotlib.pyplot as plt

data = np.load('mini_vgg_10_data.npy')
feed = np.concatenate((data[0].reshape(1,64,64,3), data[2].reshape(1,64,64,3)))
print(feed.shape)
# print(data.shape)
# plt.imshow(np.hstack((data[0], data[2])))
# plt.show()
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="model")
    return graph


graph = load_graph('./frozen_model.pb')

    # We can verify that we can access the list of operations in the graph
# for op in graph.get_operations():
#     print(op.name)

# x1 = graph.get_tensor_by_name('import/hihi/ochos1:0')
# 'Generator/truediv,Discriminator_1/real_or_fake/Sigmoid,Discriminator_1/classify_class/predict,ph_Z,istraining'
# istraining = graph.get_tensor_by_name('generator/istraining:0')
encode_ = graph.get_tensor_by_name('model/Discriminator/Encoder/dense/BiasAdd:0')
Z = graph.get_tensor_by_name('model/Placeholder_1:0')
img_1 = graph.get_tensor_by_name('model/Placeholder:0')
concat = graph.get_tensor_by_name('model/concat:0')
decode = graph.get_tensor_by_name('model/Generator/Decoder/conv2d_6/BiasAdd:0')
gen = tf.clip_by_value((decode + 1)/2, 0, 1)



#Discriminator/Encoder/dense/BiasAdd,Placeholder,Placeholder_1

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True

writer_train = tf.summary.FileWriter('./log')
with tf.Session(config=config,graph=graph) as sess: # another session
    # sess.run(tf.global_variables_initializer())
    writer_train.add_graph(sess.graph)
    output = sess.run(encode_, feed_dict={concat: feed})
    zz = np.random.uniform(-1, 1, (1,64))
    image = sess.run(gen, feed_dict={Z: np.vstack((zz, output[0].reshape(1,64)))})

    ax1 = plt.subplot(212)
    ax1.imshow(np.hstack(image))

    ax2 = plt.subplot(221)
    count, bins, ignored = ax2.hist(zz[0], 30, density=True)
    ax2.plot(bins,linewidth=2, color='r')

    ax3 = plt.subplot(222)
    count, bins, ignored = ax3.hist(output[0], 30, density=True)
    ax3.plot(bins,linewidth=2, color='b')

    plt.show()
    print('hihi', output.shape)

    
    
    
    # plt.show()
    # variables_names = [v.name for v in tf.trainable_variables()]
    # values = sess.run(variables_names)
    # for k, v in zip(variables_names, values):
    #     print("Variable: ", k)
    #     print("Shape: ", v.shape)
    #     print(v)
    
#     print(sess.run(mul, feed_dict={x1: 5}))
