from model import BEGAN
import numpy as np 

model = BEGAN()
model.build_interpolated_model()
model.init_var()