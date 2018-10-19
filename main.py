from model import BEGAN
import numpy as np 

X = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/mini_vgg_data.npy')
X = X*2-1


model = BEGAN()
model.build_interpolated_model()
model.init_var()
model.fit(X)

model.train_interpolation(X[:32],X[32:64])
