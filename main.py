from model import BEGAN
import numpy as np 

X = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/mini_vgg_data.npy')
X = X*2-1


model = BEGAN()
model.build_interpolated_model()
model.init_var()
model.fit(X)
#for i in range(500):
X_i = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/500_classes/{}.npy'.format(0))
model.train_interpolation(X_i[:50],X_i[50:], 0)
