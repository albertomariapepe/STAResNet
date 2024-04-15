from clifford import Cl, pretty
import numpy as np
import h5py
import os

pretty(precision=10)

# Dirac Algebra  `D`
D, D_blades = Cl(1,3, firstIdx=0, names='d')
locals().update(D_blades)

I = d0*d1*d2*d3

sigma1 = d1*d0
sigma2 = d2*d0
sigma3 = d3*d0


def convert_to_STA(D, H, permeability=1, permittivity=10):

    E_3d = permittivity*D
    B_3d = permeability*H

    E_4d = E_3d[:,:,:,:,:,0]*sigma1  + E_3d[:,:,:,:,:,1]*sigma2 +  E_3d[:,:,:,:,:,2]*sigma3
    B_4d = B_3d[:,:,:,:,:,0]*sigma1  + B_3d[:,:,:,:,:,1]*sigma2 +  B_3d[:,:,:,:,:,2]*sigma3


    F = E_4d + I*B_4d

    A = np.array((d01*F)(0))
    B = np.array((d02*F)(0))
    C = np.array((d03*F)(0))
    E = np.array(-(d12*F)(0))
    G = np.array(-(d13*F)(0))
    L = np.array(-(d23*F)(0))

    F = np.stack((A, B, C, E, G, L), axis=5)

    return F

traindata_path = 'data_train'
valdata_path = 'data_val'
testdata_path = 'data_test'
key_name = 'F_field'

# List all files in the folder
file_list = os.listdir(traindata_path)
counter = 0

for file in file_list:
    f = h5py.File(traindata_path+ '/' + file, 'a')
    print(traindata_path+ '/' + file)
    print(f)
    if key_name in f['train']:
        print(f"Key '{key_name}' already exists in the HDF5 file.")
        print(f['train']['F_field'])
        F = np.asarray(f['train']['F_field'])
        print(F.shape)
    else:
        print(f"Key '{key_name}' missing. Adding it now")
        # Create a new dataset (replace with your specific needs)
        D = np.asarray(f['train']['d_field'])
        H = np.asarray(f['train']['h_field'])
        F = convert_to_STA(D, H)
        F = F.astype(np.float64)
        f['train'].create_dataset(key_name, data=F)
    
    counter += 1
    print(counter, '/', len(file_list))
        

file_list = os.listdir(valdata_path)
counter = 0

for file in file_list:
    f = h5py.File(valdata_path+ '/' + file, 'a')
    print(valdata_path+ '/' + file)
    print(f)
    if key_name in f['val']:
        print(f"Key '{key_name}' already exists in the HDF5 file.")
        print(f['val']['F_field'])
    else:
        print(f"Key '{key_name}' missing. Adding it now")
        # Create a new dataset (replace with your specific needs)
        D = np.asarray(f['val']['d_field'])
        H = np.asarray(f['val']['h_field'])
        F = convert_to_STA(D, H)
        print(F.dtype)
        F = F.astype(np.float64)
        f['val'].create_dataset(key_name, data=F)
    
    counter += 1
    print(counter, '/', len(file_list))



file_list = os.listdir(testdata_path)
counter = 0

for file in file_list:
    f = h5py.File(testdata_path+ '/' + file, 'a')
    print(testdata_path+ '/' + file)
    print(f)
    if key_name in f['test']:
        print(f"Key '{key_name}' already exists in the HDF5 file.")
        print(f['test']['F_field'])
    else:
        print(f"Key '{key_name}' missing. Adding it now")
        # Create a new dataset (replace with your specific needs)
        D = np.asarray(f['test']['d_field'])
        H = np.asarray(f['test']['h_field'])
        F = convert_to_STA(D, H)
        print(F.dtype)
        F = F.astype(np.float64)
        f['test'].create_dataset(key_name, data=F)
    
    counter += 1
    print(counter, '/', len(file_list))