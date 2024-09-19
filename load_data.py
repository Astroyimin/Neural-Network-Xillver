import torch 
import torch.nn as nn
import numpy as np
import h5py
import sys

def norm_pars(pars,mins = np.array([1,0.5,5,0,3]),maxs=np.array([3.4,10,1000,4.7,87])):
    mins = mins.reshape(1,-1)
    maxs = maxs.reshape(1,-1)
    pars = (pars-mins)/(maxs-mins)
    return pars
def norm_spec(spec):
    spec = np.log(spec)
    spec = (spec -spec.min())/(spec.max()-spec.min())-1
    return spec

def transform_to_torch(X,Y,device,batchsize = 32,shuffle=True,test_split=0.01):
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")
    X = torch.tensor(X, dtype=torch.double).to(device)
    Y = torch.tensor(Y, dtype=torch.double).to(device)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataset_size = len(dataset)
    test_size = int(test_split * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=shuffle)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False) 
    return train_loader,test_loader

def load_data(path,batchsize,device,mins,maxs,normalize='log',datatype ='DataLoader'):
    with h5py.File(path, 'r') as hf:
        spec = hf['spec'][:]
        data = hf['pars'][:]
    X = norm_pars(data,mins,maxs)
    if normalize=='log':
        Y = norm_spec(spec)
        if datatype == 'DataLoader':
            train_loader,test_loader = transform_to_torch(X,Y,device,batchsize,shuffle=True)
            print('Data susscessful load')
            return train_loader,test_loader
        if datatype == 'numpy':
            print('Data susscessful load')
            return X,Y
    else:
        print('Now we just allow log normalization')
        print('Data load fail')
        sys.exit()


