from model import BEGAN
import numpy as np 

# num_classes = 500
# s_per_c = 100
# X = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/mini_vgg_data.npy')
# X = X*2-1
# Y = np.zeros((X.shape[0], num_classes))
# for i in range(num_classes):
#     Y[i*s_per_c:(i+1)*s_per_c, i] = 1

X = np.random.uniform(-1, 1, (2,64,64,3))
Y = np.zeros((2, 500))
for i in range(2):
    Y[i,i] = 1

model = BEGAN()
model.build_model()
model.init_var()
# model.load_model()
# model.fit(X, Y)
model.generate(X, Y)
#for i in range(500):
# X_i = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/500_classes/{}.npy'.format(0))
# model.train_interpolation(X_i[:50],X_i[50:], 0)
