from networks import CliffordResNet, STAResNet
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ProgbarLogger
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#tf.config.optimizer.set_jit(True)
#os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/home/alberto/anaconda3/pkgs/cuda-nvcc-12.4.99-0'
print(tf.__version__)

print("Test is GPU available: ",tf.test.is_gpu_available())

epochs = 50
seed = 1

traindata_path = 'drive/MyDrive/maxwell/data_train2D'
valdata_path = 'drive/MyDrive/maxwell/data_val'
testdata_path = 'drive/MyDrive/maxwell/data_test'

history = 2
resolution = 32
channels = 6
k = 0
hist = 2

f = h5py.File('drive/MyDrive/maxwell/data_train2D/Maxwell3D_train_0_32.h5', 'a')
print(f)
x_d = np.asarray(f['train']['d_field'][0,0,:,:,0,0])
x_h = np.asarray(f['train']['h_field'][0,0,:,:,0,0])


plt.figure()
plt.imshow(x_d, cmap='viridis')  # You can change the colormap as needed
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Image Example')
plt.savefig('fieldEx0.png')

plt.figure()
plt.imshow(x_d, cmap='viridis')  # You can change the colormap as needed
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Image Example')
plt.savefig('fieldHx0.png')


x_d = np.asarray(f['train']['d_field'][0,4,:,:,0,0])
x_h = np.asarray(f['train']['h_field'][0,4,:,:,0,0])

plt.figure()
plt.imshow(x_d, cmap='viridis')  # You can change the colormap as needed
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Image Example')
plt.savefig('fieldEx10.png')

plt.figure()
plt.imshow(x_d, cmap='viridis')  # You can change the colormap as needed
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Image Example')
plt.savefig('fieldHx10.png')


x_d = np.asarray(f['train']['d_field'][0,2,:,:,0,0])
x_h = np.asarray(f['train']['h_field'][0,2,:,:,0,0])

plt.figure()
plt.imshow(x_d, cmap='viridis')  # You can change the colormap as needed
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Image Example')
plt.savefig('fieldEx15.png')

plt.figure()
plt.imshow(x_d, cmap='viridis')  # You can change the colormap as needed
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Image Example')
plt.savefig('fieldHx15.png')

    




