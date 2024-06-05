from networks import CliffordResNet, STAResNet
import os
import tensorflow as tf
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ProgbarLogger




#tf.compat.v1.disable_eager_execution()


'''
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
tf.config.optimizer.set_jit(True)

os.environ['XLA_FLAGS']='--xla_gpu_cuda_data_dir=gpfs/home/a/albertopepe/anaconda3/pkgs/cuda-nvcc-12.4.99-0'
'''
print(tf.__version__)

print("Test is GPU available: ",tf.test.is_gpu_available())
print("available GPUs: ",tf.config.list_physical_devices('GPU'))

epochs = 50
seed = 28996

random.seed(seed)

path = os.getcwd()


traindata_path = 'data_train3D_10s'
valdata_path = 'data_val3D_10s'
testdata_path = 'data_test3D_10s'

history = 2
resolution = 28
channels = 6

print(traindata_path)



model = STAResNet(dict(shape=(resolution, resolution, resolution, history, channels)))
#model = CliffordResNet(dict(shape=(resolution, resolution, resolution, history, channels)))

lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-3, decay_steps=10000,decay_rate =0.9)
opt = tf.keras.optimizers.Adam(learning_rate = 1e-3)
model.compile(optimizer=opt,
              loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

file_list = os.listdir(traindata_path)[:100]
progbar = ProgbarLogger(count_mode = "steps")

print(model.summary())




resume = -1


if resume != -1:
    print("loading best weights...", flush= True)
    model.load_weights('models/STA_ResNet_epoch_'+str(resume) + '.h5')
    print("Done!")


training_losses = []
validation_losses = []

max_val_loss = 10e9

batchsize = 2
hist = 4

for epoch in range(resume+1,epochs):
    file_list = os.listdir(traindata_path)
    file_list.pop(0)
    
    random.shuffle(file_list)
    count = 0
    avg_loss = 0
    batches = 0

    for file in file_list:
        print(file)
        if file != '.ipynb_checkpoints':
            f = h5py.File(traindata_path+ '/' + file, 'a')
        #print(file)
        for j in range(32 - batchsize):
            for k in range(0, 9, 3):
                x_d = np.asarray(f['train']['d_field'][j:j+batchsize,k:k+2,:,:,:,:])
                x_h = np.asarray(f['train']['h_field'][j:j+batchsize:,k:k+2,:,:,:,:])

                #x_h = np.expand_dims(x_h, axis = 4)
                
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
            
                x = np.concatenate((x_d, x_h), axis=5)
            
                #x = np.expand_dims(x, axis=5)
                x = tf.Variable(initial_value=x, trainable=True, dtype=tf.float64)

                
                x = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])
                #x = tf.transpose(x, perm=[0, 2, 3, 1, 4])

                y_d = np.asarray(f['train']['d_field'][j:j+batchsize,k+2,:,:,:,:])
                y_h = np.asarray(f['train']['h_field'][j:j+batchsize,k+2,:,:,:,:])

                #y_h = np.expand_dims(y_h, axis = 3)
                y = np.concatenate((y_d, y_h), axis=-1)
                #y = np.expand_dims(y, axis=5)
                y = tf.Variable(initial_value=y, trainable=True, dtype=tf.float64)
                #y = tf.transpose(y, perm=[0, 2, 3, 4, 1, 5])

                #print(x.shape, y.shape)

                loss = model.train_on_batch(x, y)
                avg_loss += loss
                count += 1
                #progbar.update(count, [('loss', loss)])
                batches += 1
                tf.keras.backend.clear_session()

                del x, y
            
        #print(batches, avg_loss/count)

        print("epoch:", epoch, "batch:", batches, "loss:", avg_loss / count, flush=True)
       
    training_losses.append(avg_loss / count)
    print(model.optimizer.learning_rate)
        

    file_list_val = os.listdir(valdata_path)[:30]
    file_list_val.pop(0)
    random.shuffle(file_list_val)
    count = 0
    avg_val_loss = 0
    print("validating..", flush=True)
    for file in file_list_val:
        print(file)
       
        if file != '.ipynb_checkpoints':
            f = h5py.File(valdata_path+ '/' + file, 'a')

        for j in range(0, 32 - batchsize, batchsize):
            for k in range(0, 9, 3):
                print(f)
                x_d = np.asarray(f['val']['d_field'][j:j+batchsize,k:k+2,:,:,:,:])
                x_h = np.asarray(f['val']['h_field'][j:j+batchsize,k:k+2,:,:,:,:])

                #x_h = np.expand_dims(x_h, axis = 4)

                x = np.concatenate((x_d, x_h), axis=-1)
                #x = np.expand_dims(x, axis=5)

                x = tf.Variable(initial_value=x, trainable=True, dtype=tf.float64)
                #xval = tf.transpose(x, perm=[0, 2, 3, 1, 4])
                xval = tf.transpose(x, perm=[0, 2, 3, 4, 1, 5])


                y_d = np.asarray(f['val']['d_field'][j:j+batchsize,k+2,:,:,:,:])
                y_h = np.asarray(f['val']['h_field'][j:j+batchsize,k+2,:,:,:,:])

                #y_h = np.expand_dims(y_h, axis = 3)
                y = np.concatenate((y_d, y_h), axis=-1)

                #y = np.expand_dims(y, axis=4)
                yval = tf.Variable(initial_value=y, trainable=True, dtype=tf.float64)
                #yval = tf.transpose(y, perm=[0, 2, 3, 4, 1, 5])

                valloss = model.test_on_batch(xval, yval)
                avg_val_loss += valloss
                count += 1

                tf.keras.backend.clear_session()

                del x, xval, yval
            #progbar.update(count + 1, [('loss', loss)])

    validation_losses.append(avg_val_loss/count)
    print("epoch:", epoch, "val_loss:", avg_val_loss/count, flush=True)

    if avg_val_loss/count < max_val_loss:
        print("found a new best!", flush=True)
        max_val_loss = avg_val_loss/count
        model.save_weights("models/STA_ResNet_epoch_"+str(epoch)+".h5")

    # Optionally, add validation step here if needed

    # Apply callbacks
    #lr_scheduler.on_epoch_end(epoch, logs={})
    #early_stopping.on_epoch_end(epoch, logs={})

    # Check for early stopping

    print(training_losses, flush=True)
    print(validation_losses, flush=True)
    
    np.save('models/STA_training_losses_'+ str(resume)+'.npy', np.array(training_losses))
    np.save('models/STA_validation_losses_'+ str(resume)+'.npy', np.array(validation_losses))
    

epochs_range = np.arange(len(training_losses))
plt.figure(figsize=(12,6))
plt.plot(epochs_range, training_losses, label = 'Training Loss')
plt.plot(epochs_range, validation_losses, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

#plt.savefig('drive/MyDrive/maxwell/STA_losses_plot.png')


    



