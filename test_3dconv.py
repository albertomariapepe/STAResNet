from layers import RotorConv2D, RotorConv3D
import tensorflow as tf
from tfga.tfga import GeometricAlgebra
from tfga.layers import TensorToGeometric, GeometricToTensor
import numpy as numpy



sta = GeometricAlgebra([1, -1, -1, -1])
print(sta)

bv_blade_indices = sta.get_blade_indices_of_degree(degree=2)
#print(bv_blade_indices)

res = 20
bv_blade_indices =  tf.cast(bv_blade_indices, dtype=tf.int64)


tensor_shape = (2, 32, 32, 2, 6)
random_tensor = tf.random.uniform(shape=tensor_shape)
print(random_tensor.shape)

model = tf.keras.Sequential([
    TensorToGeometric(sta, blade_indices=bv_blade_indices),
    RotorConv2D(algebra = sta, stride_vertical=1, stride_horizontal=1, padding="SAME", kernel_size=3, filters =1, use_bias=False),
    GeometricToTensor(sta, blade_indices=bv_blade_indices)
])

out = model(random_tensor)
print(out.shape)



tensor_shape = (2, 32, 32, 32, 2, 6)
random_tensor = tf.random.uniform(shape=tensor_shape)
print(random_tensor.shape)

model = tf.keras.Sequential([
    TensorToGeometric(sta, blade_indices=bv_blade_indices),
    RotorConv3D(algebra = sta, stride_vertical=1, stride_depth=1, stride_horizontal=1, padding="SAME", kernel_size=3, filters =1, use_bias=False),
    GeometricToTensor(sta, blade_indices=bv_blade_indices)
])

out = model(random_tensor)
print(out.shape)
