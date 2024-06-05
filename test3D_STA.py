from networks import CliffordResNet, STAResNet
import os
import tensorflow as tf
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from clifford import Cl, pretty
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

pretty(precision=10)

D, D_blades = Cl(1,3, firstIdx=0, names='d')
locals().update(D_blades)

I = d0*d1*d2*d3

sigma1 = d1*d0
sigma2 = d2*d0
sigma3 = d3*d0

#tf.config.optimizer.set_jit(True)
#os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=/home/alberto/anaconda3/pkgs/cuda-nvcc-12.4.99-0'
print(tf.__version__)

print("Test is GPU available: ",tf.test.is_gpu_available())

epochs = 50
seed = 1


testdata_path = 'data_test3D_5s'

history = 2
resolution = 28
channels = 6

model = STAResNet(dict(shape=(resolution, resolution, resolution, history, channels)))
#model = CliffordResNet2D(dict(shape=(resolution, resolution, history, channels)))

print(model.summary())
print(testdata_path)

l = tf.keras.losses.MeanSquaredError()

def custom_loss(y, targ):
    print(y.shape, targ.shape)
    y = (y - tf.math.reduce_min(y)) / (tf.math.reduce_max(y) - tf.math.reduce_min(y))
    targ = (targ - tf.math.reduce_min(targ)) / (tf.math.reduce_max(targ) - tf.math.reduce_min(targ))
    return l(y, targ)


def calculate_ssim(y, targ):
    y = (y - tf.math.reduce_min(y)) / (tf.math.reduce_max(y) - tf.math.reduce_min(y))
    targ = (targ - tf.math.reduce_min(targ)) / (tf.math.reduce_max(targ) - tf.math.reduce_min(targ))
    return tf.image.ssim(y, targ, max_val=1.0, filter_size=5, filter_sigma=1.5, k1=0.01, k2=0.03)

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-4, decay_steps=10000,decay_rate =0.9)
opt = tf.keras.optimizers.Adam(learning_rate = lr_scheduler)
model.compile(optimizer=opt,
              loss='mse')

resume = 49
print("loading best weights...", flush= True)
model.load_weights('models/STA_ResNet_final3D_5s_28.h5')
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

def correlation_coefficient(T1, T2):
    numerator = np.mean((T1 - T1.mean()) * (T2 - T2.mean()))
    denominator = T1.std() * T2.std()
    if denominator == 0:
        return 0
    else:
        result = numerator / denominator
        return result

'''
for file in file_list:
    print(file)
    if file != '.ipynb_checkpoints':
        f = h5py.File(testdata_path+ '/' + file, 'a')
    #print(file)
    for k in range(0, 9, 3):
        x_d = np.asarray(f['test']['d_field'][:,k:k+2,:,:,0,:2])
        x_h = np.asarray(f['test']['h_field'][:,k:k+2,:,:,0,2])

        x_h = np.expand_dims(x_h, axis = 4)
        
        
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

        if count < 1000000:

            y_pred = model.predict_on_batch(x)
            
            
            corr += correlation_coefficient(np.asarray(y_pred), np.asarray(y))

            
            plt.figure()
            plt.imshow(y_d[0,:,:,0], cmap='plasma',vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_Dx.png')

            plt.figure()
            plt.imshow(y_pred[0,:,:,0], cmap='plasma',vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_Dx_pred.png')

            plt.figure()
            plt.imshow(y_d[0,:,:,1], cmap='plasma', vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_Dy.png')

            plt.figure()
            plt.imshow(y_pred[0,:,:,1], cmap='plasma', vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_Dy_pred.png')

            plt.figure()
            plt.imshow(y_h[0,:,:,0], cmap='plasma',vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_Hz.png')

            plt.figure()
            plt.imshow(y_pred[0,:,:,2], cmap='plasma',vmin = -0.005, vmax = 0.005, interpolation='bilinear')  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_Hz_pred.png')

            F= y_d[0,:,:,0]*sigma1 + y_d[0,:,:,1]*sigma2 + y_h[0,:,:,0]*d1*d2
            Fpred= y_pred[0,:,:,0]*sigma1 + y_pred[0,:,:,1]*sigma2 + y_pred[0,:,:,2]*d1*d2

            Fmag = (F**2)
            Fmag = np.asarray(Fmag)
            Fmag = Fmag.astype(np.float32)

         
            Fmagpred = (Fpred**2)
            Fmagpred = np.asarray(Fmagpred)
            Fmagpred = Fmagpred.astype(np.float32)


            min_Z = -10e-5
            max_Z = 4e-5
            plt.figure()
            cc = plt.imshow(Fmag, cmap='Spectral', vmin=min_Z, vmax=max_Z, interpolation = 'bilinear')  # You can change the colormap as needed
            plt.colorbar(cc)
            plt.savefig('drive/MyDrive/maxwell/STA_F_GT.png')
            plt.savefig('drive/MyDrive/maxwell/STA_F_GT.pdf')


            plt.figure()
            plt.imshow(Fmagpred, cmap='Spectral', vmin=min_Z, vmax=max_Z, interpolation = 'bilinear')
            plt.colorbar()  # You can change the colormap as needed
            plt.savefig('drive/MyDrive/maxwell/STA_F_pred.png')
            plt.savefig('drive/MyDrive/maxwell/STA_F_pred.pdf')
            
            
            
        
        del x, y

    print("loss:", avg_loss / count, flush=True)

testing_losses.append(avg_loss / count)

print(np.median(np.array(testing_losses)))
print(corr/count)
'''




batchsize = 2


m = 1
count = 0

x = np.linspace(0, 28, 28)
y = np.linspace(0, 28, 28)
z = np.linspace(0, 28, 28)

X, Y, Z = np.meshgrid(x, y, z)

'''
for m in range(10, 11):
    testing_losses = []
    ssim = []
    correlation = []
    for file in file_list:
        print(file)
        if file != '.ipynb_checkpoints':
            f = h5py.File(testdata_path+ '/' + file, 'a')
        #print(file)

        avg_loss = 0
        avg_ssim = 0
        corr = 0

        for j in range(32 - batchsize):
            for k in range(0, 9, 3):
                x_d = np.asarray(f['test']['d_field'][j:j+batchsize,k:k+2,:,:,:,:])
                x_h = np.asarray(f['test']['h_field'][j:j+batchsize:,k:k+2,:,:,:,:])

    
            
                x = np.concatenate((x_d, x_h), axis=5)
            
                #x = np.expand_dims(x, axis=5)
                x = tf.Variable(initial_value=x, trainable=True, dtype=tf.float64)

                
                x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])

        

            

                while k < m:
                
                    #x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
            

                    y_d = np.asarray(f['test']['d_field'][j:j+batchsize,k+2,:,:,:,:])
                    y_h = np.asarray(f['test']['h_field'][j:j+batchsize,k+2,:,:,:,:])

                    y = np.concatenate((y_d, y_h), axis=-1)
                    #y = np.expand_dims(y, axis=5)
                    y = tf.Variable(initial_value=y, trainable=True, dtype=tf.float64)

                    loss = model.test_on_batch(x, y)
                    avg_loss += loss
                    #print(j)
                    #print(avg_loss)

                    z = model.predict_on_batch(x)

                    corr += correlation_coefficient(np.asarray(z), np.asarray(y))

                    v_min = 0 
                    v_max = 1
                
                    if m == 10 and count < 10 and j == 3:

                        y = y.numpy()

                        idx = 1

                        plt.figure()
                        #z[idx,:,:,0] = (z[idx,:,:,0] - np.min(z[idx,:,:,0]))/(np.max(z[idx,:,:,0]) - np.min(z[idx,:,:,0]))
                        #z[idx,:,:,1] = (z[idx,:,:,1] - np.min(z[idx,:,:,1]))/(np.max(z[idx,:,:,1]) - np.min(z[idx,:,:,1]))
                        #z[idx,:,:,2] = (z[idx,:,:,2] - np.min(z[idx,:,:,2]))/(np.max(z[idx,:,:,2]) - np.min(z[idx,:,:,2]))
                        #y[idx,:,:,0] = (y[idx,:,:,0] - np.min(y[idx,:,:,0]))/(np.max(y[idx,:,:,0]) - np.min(y[idx,:,:,0]))
                        #y[idx,:,:,1] = (y[idx,:,:,1] - np.min(y[idx,:,:,1]))/(np.max(y[idx,:,:,1]) - np.min(y[idx,:,:,1]))
                        #y[idx,:,:,2] = (y[idx,:,:,2] - np.min(y[idx,:,:,2]))/(np.max(y[idx,:,:,2]) - np.min(y[idx,:,:,2]))

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(projection='3d')

                        F= z[idx,:,:,:,0]*sigma1 + z[idx,:,:,:,1]*sigma2 + z[idx,:,:,:,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        print(Fmag.shape)

                        Fdiff = Fmag.flatten()

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(projection='3d')
                        
                        #Fdiff[Fdiff > 0.02] = 0.02
                        norm = plt.Normalize(Fdiff.min(), Fdiff.max())
                        alpha_values = norm(Fdiff)  # Alpha values proportional to Fdiff

                        # Create a color array (RGBA) using a colormap
                        cmap = cm.Spectral

                        # Generate RGBA values from the colormap
                        colors = cmap(norm(Fdiff))

                        # Modify the alpha channel based on Fdiff
                        colors[:, 3] = alpha_values 
                        
                        
                        ax.scatter(X, Y, Z, c = colors,  s = 150)
                        #ax.plot_surface(X, Y, Z, c = Fmag.flatten(), cmap = 'Spectral', s = 150, alpha = 0.3)

                        plt.show()
                        plt.savefig('plots/STA_F' + str(m) + str(k) + '.png', bbox_inches='tight')
                        

                        Fmag_pred = Fmag

                        F= y[idx,:,:,:,0]*sigma1 + y[idx,:,:,:,1]*sigma2 + y[idx,:,:,:,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        print(Fmag.shape)
                        
                        Fdiff = Fmag.flatten()

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(projection='3d')
                        

                        norm = plt.Normalize(Fdiff.min(), Fdiff.max())
                        alpha_values = norm(Fdiff)  # Alpha values proportional to Fdiff

                        # Create a color array (RGBA) using a colormap
                        cmap = cm.Spectral

                        # Generate RGBA values from the colormap
                        colors = cmap(norm(Fdiff))

                        # Modify the alpha channel based on Fdiff
                        colors[:, 3] = alpha_values 
                        
                        
                        ax.scatter(X, Y, Z, c = colors,  s = 150)
                        #ax.plot_surface(X, Y, Z, c = Fmag.flatten(), cmap = 'Spectral', s = 150, alpha = 0.3)

                        plt.show()
                        plt.savefig('plots/GT_F' + str(m) + str(k) + '.png', bbox_inches='tight')

                        Fdiff = abs(Fmag - Fmag_pred)
                        Fdiff = Fdiff.flatten()

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(projection='3d')
                    

                        norm = plt.Normalize(Fdiff.min(), Fdiff.max())
                        alpha_values = norm(Fdiff)  # Alpha values proportional to Fdiff

                        # Create a color array (RGBA) using a colormap
                        cmap = cm.viridis

                        # Generate RGBA values from the colormap
                        colors = cmap(norm(Fdiff))

                        # Modify the alpha channel based on Fdiff
                        colors[:, 3] = alpha_values 
                        
                        ax.scatter(X, Y, Z, c = colors, s = 150)
                        plt.show()
                        plt.savefig('plots/STA_Fdiff' + str(m) + str(k) + '.png', bbox_inches='tight')





                        count += 1
                    

                    z = tf.Variable(initial_value=z, trainable=True, dtype=tf.float64)

                    avg_ssim += calculate_ssim(y, z)
                    #print(avg_ssim)
                    #print("***")

                    x = tf.expand_dims(x[:,:,:,:,1,:], axis = 4)
                    z = tf.expand_dims(z, axis = 4)
                
                    x = tf.concat([x, z], 4)


                    k+= 1


        #print("loss:", avg_loss / (m), flush=True)
    testing_losses.append(avg_loss / (m*16))
    ssim.append(avg_ssim / (m*16))
    correlation.append(corr /(m*16))
    print("median error for m =", m, ":", np.median(np.array(testing_losses)))
    #print("median correlation for m =", m, ":", np.median(np.array(correlation)))

    #print("median SSIM for m =", m, ":", np.median(np.array(ssim)))


'''

for m in range(10, 11):
    testing_losses = []
    ssim = []
    correlation = []
    for file in file_list:
        print(file)
        if file != '.ipynb_checkpoints':
            f = h5py.File(testdata_path+ '/' + file, 'a')
        #print(file)

        avg_loss = 0
        avg_ssim = 0
        corr = 0

        for j in range(32 - batchsize):
            for k in range(0, 9, 3):
                x_d = np.asarray(f['test']['d_field'][j:j+batchsize,k:k+2,:,:,:,:])
                x_h = np.asarray(f['test']['h_field'][j:j+batchsize:,k:k+2,:,:,:,:])

    
            
                x = np.concatenate((x_d, x_h), axis=5)
            
                #x = np.expand_dims(x, axis=5)
                x = tf.Variable(initial_value=x, trainable=True, dtype=tf.float64)

                
                x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])

        

            

                while k < m:
                
                    #x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
            

                    y_d = np.asarray(f['test']['d_field'][j:j+batchsize,k+2,:,:,:,:])
                    y_h = np.asarray(f['test']['h_field'][j:j+batchsize,k+2,:,:,:,:])

                    y = np.concatenate((y_d, y_h), axis=-1)
                    #y = np.expand_dims(y, axis=5)
                    y = tf.Variable(initial_value=y, trainable=True, dtype=tf.float64)

                    loss = model.test_on_batch(x, y)
                    avg_loss += loss
                    #print(j)
                    #print(avg_loss)

                    z = model.predict_on_batch(x)

                    corr += correlation_coefficient(np.asarray(z), np.asarray(y))

                    v_min = 0 
                    v_max = 1
                
                    if m == 10 and count < 10 and j == 3:

                        y = y.numpy()

                        idx = 1

                        plt.figure()
                        #z[idx,:,:,0] = (z[idx,:,:,0] - np.min(z[idx,:,:,0]))/(np.max(z[idx,:,:,0]) - np.min(z[idx,:,:,0]))
                        #z[idx,:,:,1] = (z[idx,:,:,1] - np.min(z[idx,:,:,1]))/(np.max(z[idx,:,:,1]) - np.min(z[idx,:,:,1]))
                        #z[idx,:,:,2] = (z[idx,:,:,2] - np.min(z[idx,:,:,2]))/(np.max(z[idx,:,:,2]) - np.min(z[idx,:,:,2]))
                        #y[idx,:,:,0] = (y[idx,:,:,0] - np.min(y[idx,:,:,0]))/(np.max(y[idx,:,:,0]) - np.min(y[idx,:,:,0]))
                        #y[idx,:,:,1] = (y[idx,:,:,1] - np.min(y[idx,:,:,1]))/(np.max(y[idx,:,:,1]) - np.min(y[idx,:,:,1]))
                        #y[idx,:,:,2] = (y[idx,:,:,2] - np.min(y[idx,:,:,2]))/(np.max(y[idx,:,:,2]) - np.min(y[idx,:,:,2]))

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(projection='3d')

                        F= z[idx,:,:,0,0]*sigma1 + z[idx,:,:,0,1]*sigma2 + z[idx,:,:,0,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        plt.figure()
                        cc = plt.imshow(Fmag, cmap='Spectral', interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/STA_F_z0_' + str(m) + str(k) + '.png')

                        Fmag_pred_z0 = Fmag

                        F= z[idx,:,:,13,0]*sigma1 + z[idx,:,:,13,1]*sigma2 + z[idx,:,:,13,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        plt.figure()
                        cc = plt.imshow(Fmag, cmap='Spectral', interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/STA_F_z13_' + str(m) + str(k) + '.png')

                        Fmag_pred_z13 = Fmag

                        F= z[idx,:,:,27,0]*sigma1 + z[idx,:,:,27,1]*sigma2 + z[idx,:,:,27,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        plt.figure()
                        cc = plt.imshow(Fmag, cmap='Spectral', interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/STA_F_z27_' + str(m) + str(k) + '.png')

                        Fmag_pred_z27 = Fmag


                        F= y[idx,:,:,0,0]*sigma1 + y[idx,:,:,0,1]*sigma2 + y[idx,:,:,0,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        Fmag_z0 = Fmag

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        cc = plt.imshow(Fmag, cmap='Spectral', interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/GT_F_z0_' + str(m) + str(k) + '.png')

                        F= y[idx,:,:,13,0]*sigma1 + y[idx,:,:,13,1]*sigma2 + y[idx,:,:,13,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        Fmag_z13 = Fmag

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        cc = plt.imshow(Fmag, cmap='Spectral', interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/GT_F_z13_' + str(m) + str(k) + '.png')

                        F= y[idx,:,:,27,0]*sigma1 + y[idx,:,:,27,1]*sigma2 + y[idx,:,:,27,2]*sigma3

                        Fmag = (F**2)
                        Fmag = np.asarray(Fmag)
                        Fmag = Fmag.astype(np.float32)

                        Fmag_z27 = Fmag

                        #Fmag = (Fmag - np.min(Fmag)) / (np.max(Fmag) - np.min(Fmag))
                        cc = plt.imshow(Fmag, cmap='Spectral', interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/GT_F_z27_' + str(m) + str(k) + '.png')

            
                        print(np.min(abs(Fmag_z0 - Fmag_pred_z0)), np.max(abs(Fmag_z0 - Fmag_pred_z0)))
                        cc = plt.imshow(abs(Fmag_z0 - Fmag_pred_z0), cmap='viridis',  interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/STA_Fdiff_z0_' + str(m) + str(k) + '.png')

                        cc = plt.imshow(abs(Fmag_z13 - Fmag_pred_z13), cmap='viridis',  interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/STA_Fdiff_z13_' + str(m) + str(k) + '.png')

                        cc = plt.imshow(abs(Fmag_z27 - Fmag_pred_z27), cmap='viridis',  interpolation = 'bilinear')  # You can change the colormap as needed
                        plt.savefig('plots/STA_Fdiff_z27_' + str(m) + str(k) + '.png')


                 





                        count += 1
                    

                    z = tf.Variable(initial_value=z, trainable=True, dtype=tf.float64)

                    avg_ssim += calculate_ssim(y, z)
                    #print(avg_ssim)
                    #print("***")

                    x = tf.expand_dims(x[:,:,:,:,1,:], axis = 4)
                    z = tf.expand_dims(z, axis = 4)
                
                    x = tf.concat([x, z], 4)


                    k+= 1


        #print("loss:", avg_loss / (m), flush=True)
    testing_losses.append(avg_loss / (m*16))
    ssim.append(avg_ssim / (m*16))
    correlation.append(corr /(m*16))
    print("median error for m =", m, ":", np.median(np.array(testing_losses)))
    #print("median correlation for m =", m, ":", np.median(np.array(correlation)))

    #print("median SSIM for m =", m, ":", np.median(np.array(ssim)))







