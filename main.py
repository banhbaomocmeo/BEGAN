from model import BEGAN
import numpy as np 

X = np.load('/media/HDD-2T/data/VGG-Faces/VGG-Faces-ThanhTM/mini_vgg_data.npy')
X = X*2-1


model = BEGAN()
model.fit(X)