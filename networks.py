import tensorflow as tf
from tensorflow.keras import (activations, constraints, initializers, layers,
                              regularizers)

from tfga.layers import GeometricAlgebraLayer
from tfga.tfga import GeometricAlgebra
from layers import RotorConv3D, RotorConv2D
from tfga.layers import TensorToGeometric, GeometricToTensor, GeometricProductDense
from tensorflow import keras

def CliffordResNet(inputs_dict, kernel_size = 3):

    ga = GeometricAlgebra([1, 1, 1])

    vec_biv_indices = [1, 2, 3, 4, 5, 6]
    
    all_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    l_input = keras.layers.Input(**inputs_dict)

    l2_input = TensorToGeometric(ga, blade_indices=vec_biv_indices)(l_input)

    print(l2_input)

    block_0 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, stride_depth = 1, padding='SAME')(l2_input)
    
   

    block_0 = keras.layers.ReLU()(block_0)



    l_conv1 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1, padding='SAME')(block_0)
    l_conv1 = keras.layers.ReLU()(l_conv1)
    l_conv2 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, stride_depth = 1,padding='SAME')(l_conv1)
    l_skip1 = keras.layers.add([block_0, l_conv2])
    block_1 = keras.layers.ReLU()(l_skip1)

    l_conv3 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_1)
    l_conv3 = keras.layers.ReLU()(l_conv3)
    l_conv4 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv3)
    l_skip2 = keras.layers.add([block_1, l_conv4])
    block_2 = keras.layers.ReLU()(l_skip2)

    l_conv5 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_2)
    l_conv5 = keras.layers.ReLU()(l_conv5)
    l_conv6 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv5)
    l_skip3 = keras.layers.add([block_2, l_conv6])
    block_3 = keras.layers.ReLU()(l_skip3)

    l_conv7 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_3)
    l_conv7 = keras.layers.ReLU()(l_conv7)
    l_conv8 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv7)
    l_skip4 = keras.layers.add([block_3, l_conv8])
    block_4 = keras.layers.ReLU()(l_skip4)

    l_conv9 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_4)
    l_conv9 = keras.layers.ReLU()(l_conv9)
    l_conv10 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv9)
    l_skip5 = keras.layers.add([block_4, l_conv10])
    block_5 = keras.layers.ReLU()(l_skip5)
    
    l_conv11 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_5)
    l_conv11 = keras.layers.ReLU()(l_conv11)
    l_conv12 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv11)
    l_skip6 = keras.layers.add([block_5, l_conv12])
    block_6 = keras.layers.ReLU()(l_skip6)    
    
    l_conv13 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_6)
    l_conv13 = keras.layers.ReLU()(l_conv13)
    l_conv14 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv13)
    l_skip7 = keras.layers.add([block_6, l_conv14])
    block_7 = keras.layers.ReLU()(l_skip7)   
    
    l_conv15 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_7)
    l_conv15 = keras.layers.ReLU()(l_conv15)
    l_conv16 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv15)
    l_skip8 = keras.layers.add([block_7, l_conv16])
    block_8 = keras.layers.ReLU()(l_skip8) 
   
    l_conv17 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size = kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_8)
    l_conv17 = keras.layers.ReLU()(l_conv17)
    l_conv18 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv17)
    l_skip9 = keras.layers.add([block_8, l_conv18])
    block_9 = keras.layers.ReLU()(l_skip9) 
    
    l_conv19 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = vec_biv_indices, filters=11, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_9)
    l_conv19 = keras.layers.ReLU()(l_conv19)
    l_conv20 = RotorConv3D(algebra = ga, blade_indices_kernel =all_indices,
            blade_indices_bias = all_indices, filters=11, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv19)
    l_skip10 = keras.layers.add([block_9, l_conv20])
    block_10 = keras.layers.ReLU()(l_skip10)
    
    l_output = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=1,  kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_10)
    

    print("output:", l_output)
    l_output = GeometricToTensor(algebra = ga, blade_indices=vec_biv_indices)(l_output)
    print(l_output)
    l2_output = tf.reshape(l_output, (-1, l_output.shape[1], l_output.shape[2],l_output.shape[2], 6))
    print(l2_output)
    return keras.models.Model(inputs=l_input, outputs=l2_output, name='CliffordResNet')    


def STAResNet(inputs_dict, kernel_size = 3):

    ga = GeometricAlgebra([1, -1, -1, -1])
    vec_biv_indices = [5, 6, 7, 8, 9, 10]
    all_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    l_input = keras.layers.Input(**inputs_dict)

    print(l_input)

    l2_input = TensorToGeometric(ga, blade_indices=vec_biv_indices)(l_input)

    print(l2_input)

    block_0 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, stride_depth = 1,padding='SAME')(l2_input)
    
    print(block_0)
    
    block_0 = keras.layers.ReLU()(block_0)

    print(block_0)

    l_conv1 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_0)
    l_conv1 = keras.layers.ReLU()(l_conv1)
    l_conv2 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv1)
    l_skip1 = keras.layers.add([block_0, l_conv2])
    block_1 = keras.layers.ReLU()(l_skip1)

    l_conv3 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, stride_depth = 1, padding='SAME')(block_1)
    l_conv3 = keras.layers.ReLU()(l_conv3)
    l_conv4 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, stride_depth = 1, padding='SAME')(l_conv3)
    l_skip2 = keras.layers.add([block_1, l_conv4])
    block_2 = keras.layers.ReLU()(l_skip2)

    l_conv5 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_2)
    l_conv5 = keras.layers.ReLU()(l_conv5)
    l_conv6 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv5)
    l_skip3 = keras.layers.add([block_2, l_conv6])
    block_3 = keras.layers.ReLU()(l_skip3)

    l_conv7 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_3)
    l_conv7 = keras.layers.ReLU()(l_conv7)
    l_conv8 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv7)
    l_skip4 = keras.layers.add([block_3, l_conv8])
    block_4 = keras.layers.ReLU()(l_skip4)

    l_conv9 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_4)
    l_conv9 = keras.layers.ReLU()(l_conv9)
    l_conv10 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv9)
    l_skip5 = keras.layers.add([block_4, l_conv10])
    block_5 = keras.layers.ReLU()(l_skip5)
    
    l_conv11 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_5)
    l_conv11 = keras.layers.ReLU()(l_conv11)
    l_conv12 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv11)
    l_skip6 = keras.layers.add([block_5, l_conv12])
    block_6 = keras.layers.ReLU()(l_skip6)    
    
    l_conv13 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_6)
    l_conv13 = keras.layers.ReLU()(l_conv13)
    l_conv14 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv13)
    l_skip7 = keras.layers.add([block_6, l_conv14])
    block_7 = keras.layers.ReLU()(l_skip7)   
    
    l_conv15 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_7)
    l_conv15 = keras.layers.ReLU()(l_conv15)
    l_conv16 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv15)
    l_skip8 = keras.layers.add([block_7, l_conv16])
    block_8 = keras.layers.ReLU()(l_skip8) 
   
    l_conv17 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size = kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_8)
    l_conv17 = keras.layers.ReLU()(l_conv17)
    l_conv18 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv17)
    l_skip9 = keras.layers.add([block_8, l_conv18])
    block_9 = keras.layers.ReLU()(l_skip9) 
    
    l_conv19 = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = vec_biv_indices, filters=8, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_9)
    l_conv19 = keras.layers.ReLU()(l_conv19)
    l_conv10 = RotorConv3D(algebra = ga, blade_indices_kernel =all_indices,
            blade_indices_bias = all_indices, filters=8, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(l_conv19)
    l_skip10 = keras.layers.add([block_9, l_conv10])
    block_10 = keras.layers.ReLU()(l_skip10)
    
    l_output = RotorConv3D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=1,  kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,stride_depth = 1,padding='SAME')(block_10)
    

    
    l_output = GeometricToTensor(algebra = ga, blade_indices=vec_biv_indices)(l_output)
    print(l_output)
    l2_output = tf.reshape(l_output, (-1, l_output.shape[1], l_output.shape[2],l_output.shape[2],  6))
    print(l2_output)
    return keras.models.Model(inputs=l_input, outputs=l2_output, name='STAResNet')  


filters = 32

def CliffordResNet2D(inputs_dict, kernel_size = 3):

    ga = GeometricAlgebra([1, 1])

    vec_biv_indices = [1, 2, 3]
    
    all_indices = [0, 1, 2, 3]

    l_input = keras.layers.Input(**inputs_dict)

    l2_input = TensorToGeometric(ga, blade_indices=vec_biv_indices)(l_input)

    print(l2_input)

    block_0 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l2_input)
    block_0 = keras.layers.ReLU()(block_0)

    l_conv1 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_0)
    l_conv1 = keras.layers.ReLU()(l_conv1)
    l_conv2 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv1)
    l_skip1 = keras.layers.add([block_0, l_conv2])
    block_1 = keras.layers.ReLU()(l_skip1)

    l_conv3 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_1)
    l_conv3 = keras.layers.ReLU()(l_conv3)
    l_conv4 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv3)
    l_skip2 = keras.layers.add([block_1, l_conv4])
    block_2 = keras.layers.ReLU()(l_skip2)

    l_conv5 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_2)
    l_conv5 = keras.layers.ReLU()(l_conv5)
    l_conv6 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv5)
    l_skip3 = keras.layers.add([block_2, l_conv6])
    block_3 = keras.layers.ReLU()(l_skip3)

    l_conv7 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_3)
    l_conv7 = keras.layers.ReLU()(l_conv7)
    l_conv8 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv7)
    l_skip4 = keras.layers.add([block_3, l_conv8])
    block_4 = keras.layers.ReLU()(l_skip4)

    l_conv9 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_4)
    l_conv9 = keras.layers.ReLU()(l_conv9)
    l_conv10 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv9)
    l_skip5 = keras.layers.add([block_4, l_conv10])
    block_5 = keras.layers.ReLU()(l_skip5)
    
    l_conv11 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_5)
    l_conv11 = keras.layers.ReLU()(l_conv11)
    l_conv12 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv11)
    l_skip6 = keras.layers.add([block_5, l_conv12])
    block_6 = keras.layers.ReLU()(l_skip6)    
    
    l_conv13 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_6)
    l_conv13 = keras.layers.ReLU()(l_conv13)
    l_conv14 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv13)
    l_skip7 = keras.layers.add([block_6, l_conv14])
    block_7 = keras.layers.ReLU()(l_skip7)   
    
    l_conv15 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_7)
    l_conv15 = keras.layers.ReLU()(l_conv15)
    l_conv16 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv15)
    l_skip8 = keras.layers.add([block_7, l_conv16])
    block_8 = keras.layers.ReLU()(l_skip8) 
   
    l_conv17 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size = kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_8)
    l_conv17 = keras.layers.ReLU()(l_conv17)
    l_conv18 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv17)
    l_skip9 = keras.layers.add([block_8, l_conv18])
    block_9 = keras.layers.ReLU()(l_skip9) 
    
    l_conv19 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = vec_biv_indices, filters=filters, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_9)
    l_conv19 = keras.layers.ReLU()(l_conv19)
    l_conv20 = RotorConv2D(algebra = ga, blade_indices_kernel =all_indices,
            blade_indices_bias = all_indices, filters=filters, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv19)
    l_skip10 = keras.layers.add([block_9, l_conv20])
    block_10 = keras.layers.ReLU()(l_skip10)
    
    l_output = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=1,  kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_10)
    

    print(l_output)
    l_output = GeometricToTensor(algebra = ga, blade_indices=vec_biv_indices)(l_output)
    print(l_output)
    l2_output = tf.reshape(l_output, (-1, l_output.shape[1], l_output.shape[2], 3))
    print(l2_output)
    return keras.models.Model(inputs=l_input, outputs=l2_output, name='CliffordResNet')    



filters_STA = 24



def STAResNet2D(inputs_dict, kernel_size = 3):

    ga = GeometricAlgebra([1, -1, -1])
    vec_biv_indices = [4, 5, 6]
    all_indices = [0, 1, 2, 3, 4, 5, 6, 7]

    l_input = keras.layers.Input(**inputs_dict)

    print(l_input)

    l2_input = TensorToGeometric(ga, blade_indices=vec_biv_indices)(l_input)

    print(l2_input)

    block_0 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l2_input)
    block_0 = keras.layers.ReLU()(block_0)

    l_conv1 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_0)
    l_conv1 = keras.layers.ReLU()(l_conv1)
    l_conv2 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv1)
    l_skip1 = keras.layers.add([block_0, l_conv2])
    block_1 = keras.layers.ReLU()(l_skip1)

    l_conv3 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,  padding='SAME')(block_1)
    l_conv3 = keras.layers.ReLU()(l_conv3)
    l_conv4 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv3)
    l_skip2 = keras.layers.add([block_1, l_conv4])
    block_2 = keras.layers.ReLU()(l_skip2)

    l_conv5 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_2)
    l_conv5 = keras.layers.ReLU()(l_conv5)
    l_conv6 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(l_conv5)
    l_skip3 = keras.layers.add([block_2, l_conv6])
    block_3 = keras.layers.ReLU()(l_skip3)

    l_conv7 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1, padding='SAME')(block_3)
    l_conv7 = keras.layers.ReLU()(l_conv7)
    l_conv8 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1 ,padding='SAME')(l_conv7)
    l_skip4 = keras.layers.add([block_3, l_conv8])
    block_4 = keras.layers.ReLU()(l_skip4)

    l_conv9 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1 ,padding='SAME')(block_4)
    l_conv9 = keras.layers.ReLU()(l_conv9)
    l_conv10 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv9)
    l_skip5 = keras.layers.add([block_4, l_conv10])
    block_5 = keras.layers.ReLU()(l_skip5)
    
    l_conv11 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1 ,padding='SAME')(block_5)
    l_conv11 = keras.layers.ReLU()(l_conv11)
    l_conv12 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1 ,padding='SAME')(l_conv11)
    l_skip6 = keras.layers.add([block_5, l_conv12])
    block_6 = keras.layers.ReLU()(l_skip6)    
    
    l_conv13 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_6)
    l_conv13 = keras.layers.ReLU()(l_conv13)
    l_conv14 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv13)
    l_skip7 = keras.layers.add([block_6, l_conv14])
    block_7 = keras.layers.ReLU()(l_skip7)   
    
    l_conv15 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_7)
    l_conv15 = keras.layers.ReLU()(l_conv15)
    l_conv16 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv15)
    l_skip8 = keras.layers.add([block_7, l_conv16])
    block_8 = keras.layers.ReLU()(l_skip8) 
   
    l_conv17 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size = kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_8)
    l_conv17 = keras.layers.ReLU()(l_conv17)
    l_conv18 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv17)
    l_skip9 = keras.layers.add([block_8, l_conv18])
    block_9 = keras.layers.ReLU()(l_skip9) 
    
    l_conv19 = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = vec_biv_indices, filters=filters_STA, kernel_size= kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_9)
    l_conv19 = keras.layers.ReLU()(l_conv19)
    l_conv10 = RotorConv2D(algebra = ga, blade_indices_kernel =all_indices,
            blade_indices_bias = all_indices, filters=filters_STA, kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(l_conv19)
    l_skip10 = keras.layers.add([block_9, l_conv10])
    block_10 = keras.layers.ReLU()(l_skip10)
    
    l_output = RotorConv2D(algebra = ga, blade_indices_kernel = all_indices,
            blade_indices_bias = all_indices, filters=1,  kernel_size=kernel_size, stride_horizontal = 1, stride_vertical = 1,padding='SAME')(block_10)
    

    
    l_output = GeometricToTensor(algebra = ga, blade_indices=vec_biv_indices)(l_output)
    print(l_output)
    l2_output = tf.reshape(l_output, (-1, l_output.shape[1], l_output.shape[2],  3))
    print(l2_output)
    return keras.models.Model(inputs=l_input, outputs=l2_output, name='STAResNet')  