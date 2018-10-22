from model import BEGAN
import numpy as np 

# num_classes = 500
# s_per_c = 100
# X = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/mini_vgg_data.npy')
# X = X*2-1
# Y = np.zeros((X.shape[0], num_classes))
# for i in range(num_classes):
#     Y[i*s_per_c:(i+1)*s_per_c, i] = 1

model = BEGAN()
# model.build_interpolated_model()
model.init_var()
# model.fit(X, Y)
#for i in range(500):
# X_i = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/500_classes/{}.npy'.format(0))
# model.train_interpolation(X_i[:50],X_i[50:], 0)
