from model import BEGAN
import numpy as np 
from ultis import number_to_list_bin

num_classes = 500
s_per_c = 100
X = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/mini_vgg_data.npy')
X = X*2-1
# Y = np.zeros((X.shape[0], num_classes))
# for i in range(num_classes):
#     Y[i*s_per_c:(i+1)*s_per_c, i] = 1

# X = np.random.uniform(-1, 1, (2,64,64,3))
# Y = np.zeros((2, 500))
length = int(np.ceil(np.log2(500)))
print('length: ', length)

Y = np.array([s_per_c*[number_to_list_bin(num, length)] for num in range(500)]).reshape((-1, length))
print(Y)
print(Y.shape)
model = BEGAN(num_classes=length)
model.build_model()
model.init_var()
# model.load_model()
# model.fit(X, Y)

i_g = np.random.choice(1000, 100)
X_g = X[i_g]
Y_g = Y[i_g]
model.generate(X_g, Y_g)
