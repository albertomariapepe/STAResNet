from networks import CliffordResNet2D, STAResNet2D
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ProgbarLogger
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import clifford
from clifford import Cl, pretty

pretty(precision=10)

D, D_blades = Cl(1,2, firstIdx=0, names='d')
locals().update(D_blades)

I = d0*d1*d2

sigma1 = d1*d0
sigma2 = d2*d0

#tf.config.optimizer.set_jit(True)
#os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/home/alberto/anaconda3/pkgs/cuda-nvcc-12.4.99-0'
print(tf.__version__)

print("Test is GPU available: ",tf.test.is_gpu_available())

epochs = 49
seed = 1


testdata_path = 'drive/MyDrive/maxwell/data_test2D_obst'

history = 2
resolution = 48
channels = 3

#model = STAResNet2D(dict(shape=(resolution, resolution, history, channels)))
model = CliffordResNet2D(dict(shape=(resolution, resolution, history, channels)))

print(model.summary())

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-4, decay_steps=10000,decay_rate =0.9)
opt = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)
model.compile(optimizer=opt,
              loss='mse')

resume = 46
print("loading best weights...", flush= True)
model.load_weights('drive/MyDrive/maxwell/GA_ResNet_epoch_'+str(resume) + '.h5')
#model.load_weights('GA_ResNet_epoch'+str(resume) + '.h5')


file_list = os.listdir(testdata_path)
file_list.sort()
print(file_list)
#file_list = file_list[10:]
print(file_list)

print("Done!")

testing_losses = []
avg_loss = 0
count = 0
corr = 0

from scipy import signal

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result



for file in file_list:
    print(file)
    if file != '.ipynb_checkpoints':
        f = h5py.File(testdata_path+ '/' + file, 'a')
    
    for k in range(0, 9, 3):
        x_d = np.asarray(f['test']['d_field'][:,k:k+2,:,:,0,:2])
        x_h = np.asarray(f['test']['h_field'][:,k:k+2,:,:,0,2])

        x_h = np.expand_dims(x_h, axis = 4)
        
        '''
        plt.figure()
        plt.imshow(x_d[0,0,:,:,0,0], cmap='viridis')  # You can change the colormap as needed
        plt.colorbar()  # Add a colorbar for reference
        plt.title('2D Image Example')
        plt.savefig('Dxyx.png')

        plt.figure()
        plt.imshow(x_d[0,0,:,:,0,1], cmap='viridis')  # You can change the colormap as needed
        plt.colorbar()  # Add a colorbar for reference
        plt.title('2D Image Example')
        plt.savefig('Dxyy.png')

        plt.figure()
        plt.imshow(x_d[0,0,:,:,0,2], cmap='viridis')  # You can change the colormap as needed
        plt.colorbar()  # Add a colorbar for reference
        plt.title('2D Image Example')
        plt.savefig('Dxyz.png')
        '''
        x = np.concatenate((x_d, x_h), axis=-1)
        x = tf.Variable(initial_value=x, trainable=True, dtype=tf.float64)
        
        #x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
        x = tf.transpose(x, perm=[0, 2, 3, 1, 4])

        y_d = np.asarray(f['test']['d_field'][:,k+2,:,:,0,:2])
        y_h = np.asarray(f['test']['h_field'][:,k+2,:,:,0,2])

        y_h = np.expand_dims(y_h, axis = 3)
        y = np.concatenate((y_d, y_h), axis=-1)
        #y = np.expand_dims(y, axis=4)
        y = tf.Variable(initial_value=y, trainable=True, dtype=tf.float64)
        #y = tf.transpose(y, perm=[0, 2, 3, 4, 1, 5])

        loss = model.test_on_batch(x, y)
        avg_loss += loss
        count += 1

        if count < 100000:

            y_pred = model.predict_on_batch(x)

            corr += correlation_coefficient(np.asarray(y_pred), np.asarray(y))

            '''
            plt.figure()
            #print(np.min(y_d[0,:,:,0]), np.max(y_d[0,:,:,0]))
            plt.imshow(y_d[0,:,:,0], cmap='plasma', vmin = -0.005, vmax = 0.005,  interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_Dx.png')

            plt.figure()
            plt.figure()
            plt.imshow(y_pred[0,:,:,0], cmap='plasma', vmin = -0.005, vmax = 0.005,  interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_Dx_pred.png')

            plt.figure()
            plt.imshow(y_d[0,:,:,1], cmap='plasma', vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_Dy.png')

            plt.figure()
            plt.figure()
            plt.imshow(y_pred[0,:,:,1], cmap='plasma', vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_Dy_pred.png')

            plt.figure()
            plt.imshow(y_h[0,:,:,0], cmap='plasma', vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_Hz.png')

            plt.figure()
            plt.figure()
            plt.imshow(y_pred[0,:,:,2], cmap='plasma', vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_Hz_pred.png')

            F= y_d[0,:,:,0]*sigma1 + y_d[0,:,:,1]*sigma2 + y_h[0,:,:,0]*d1*d2
            Fpred= y_pred[0,:,:,0]*sigma1 + y_pred[0,:,:,1]*sigma2 + y_pred[0,:,:,2]*d1*d2

            Fmag = (F**2)
            Fmag = np.asarray(Fmag)
            Fmag = Fmag.astype(np.float32)

         
            Fmagpred = (Fpred**2)
            Fmagpred = np.asarray(Fmagpred)
            Fmagpred = Fmagpred.astype(np.float32)

            print(np.max(Fmag), print(np.min(Fmag)))
            


            min_Z = -10e-5
            max_Z = 4e-5
            plt.figure()
            cc = plt.imshow(Fmag, cmap='Spectral', vmin=min_Z, vmax=max_Z, interpolation = 'bilinear')  # You can change the colormap as needed
            plt.colorbar(cc)
            plt.savefig('drive/MyDrive/maxwell/GA_F_GT.png')
            plt.savefig('drive/MyDrive/maxwell/GA_F_GT.pdf')


            plt.figure()
            plt.imshow(Fmagpred, cmap='Spectral', vmin=min_Z, vmax=max_Z, interpolation = 'bilinear')
            plt.colorbar()  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/GA_F_pred.png')
            plt.savefig('drive/MyDrive/maxwell/GA_F_pred.pdf')

            '''
            
            
            
        
        del x, y

    print("loss:", avg_loss / count, flush=True)

testing_losses.append(avg_loss / count)

print(np.median(np.array(testing_losses)))
print(corr/count)




    




