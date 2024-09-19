import h5py
import numpy as np
import sys
import torch
import time
import model as ml
import matplotlib.pyplot as plt
import matplotlib
import xspec
import pandas as pd
import os
# plot default
plt.rcParams['text.usetex'] = False

# random seed
np.random.seed(42)

# Xspec default
xspec.AllModels.lmod('relxill','/data/models/relxill_v2.3')
xspec.AllModels.setEnergies(f"0.0717702 1000.1 3000 log")

# get time 
c1time = time.time()

def norm_pars(pars,mins = np.array([1,0.5,5,0,3]),maxs=np.array([3.4,10,1000,4.7,87])):
    """
        normalize parameters
    """
    mins = mins.reshape(1,-1)
    maxs = maxs.reshape(1,-1)
    pars = (pars-mins)/(maxs-mins)
    return pars

def norm_spec(spec):
    """
        Normalize spectrum
    """
    spec = np.log(spec)
    spec = (spec -spec.min())/(spec.max()-spec.min())-1
    return spec

def load_data(path,mins,maxs,normalize='log',sample=1000):
    """
        Load data 
    """
    with h5py.File(path, 'r') as hf:
        spec = hf['spec'][:]
        data = hf['pars'][:]
    X = norm_pars(data,mins,maxs)
    if normalize=='log':
        Y = norm_spec(spec)
        indices = np.random.choice(X.shape[0], size=sample, replace=False)
        X = X[indices]
        Y = Y[indices]
        input_tensor = torch.from_numpy(X).double()
        return input_tensor,Y   
    else:
        print('Now we just allow log normalization')
        print('Data load fail')
        sys.exit()

def denorm_pars(normalized_pars, mins = np.array([1,0.5,5,0,3]),maxs=np.array([3.4,10,1000,4.7,87])):
    """
        renorm parameters
    """
    # Reshape mins and maxs to ensure they are broadcasted correctly with normalized_pars
    mins = mins.reshape(1, -1)
    maxs = maxs.reshape(1, -1)
    # Perform the denormalization
    original_pars = normalized_pars * (maxs - mins) + mins
    return original_pars

def denorm_spec(x):
    """
        renorm spectrum
    """
    x = x+1 
    return np.exp(x)

def load_xspec_model(pars):
    """
    Get spectrum from xspec
    args:
        pars: xillver parameter
    """
    Name = 'xillver'
    path = 'tem.dat'
    if os.path.exists(path):
        os.remove(path)
    model = xspec.Model(Name)
    model.xillver.gamma.values = pars[0]
    model.xillver.Afe.values =pars[1]
    model.xillver.Ecut.values = pars[2]
    model.xillver.logxi.values = pars[3]
    model.xillver.Incl.values = pars[4]
    xspec.Plot.device = '/null'
    xspec.Plot.add = True
    xspec.Plot.addCommand(f'wd {path}')
    xspec.Plot('eemodel') 
    xspec.Plot.addCommand(f'exit')
    xspec.Plot.commands = ()
    xspec.AllModels.clear()
    df = pd.read_csv(path, sep='\s+', header=None, skiprows=3)
    spec = np.array(df[2])
    # Here Just confirm the XSPEC spec has same normalization process as training data
    spec = denorm_spec(norm_spec(spec))
    return spec

# Load data
numspec = 1000
inputpars,spec= load_data(path ='/data/home/yimin2/advanced_model/DataSet/DataSet100000_paper.h5',
                                   mins = np.array([1.,0.5,5,0.,3]),
                                   maxs = np.array([3.4,10,1000,4.7,87]),
                                   normalize='log',
                                   sample = numspec)

# Get the cpu
device = torch.device('cpu')

# load model and move to cpu
Gener = ml.Generator().to(device)
inputpars =inputpars.to(device)
Gener.load_state_dict(torch.load('model_dict.pth',map_location=torch.device('cpu')))

# to float64
Gener.double()

# Print necessary information, make sure the data type is float64
print(f"Input tensor dtype: {inputpars.dtype}")
for name, param in Gener.named_parameters():
    print(f"Parameter {name} has dtype {param.dtype}")

# Comfirm the model and data are load on cpu
print(f"Input tensor is on device: {inputpars.device}")
device = next(Gener.parameters()).device
print(f"Model is on device: {device}")

# Get spectrum
with torch.no_grad():
    ctime =time.time()
    outputs = Gener(inputpars)
print(f'Use time for generate {numspec} spectrum:',time.time()-ctime)

# To numpy array
outputs = outputs.cpu().detach().numpy()
inputpars = inputpars.cpu().detach().numpy()

# get sample for plot test
sample_plot = 10
indices = np.random.choice(outputs.shape[0], size=sample_plot, replace=False)

# get sample data
outputs_sampled = outputs[indices]
inputpars_sampled = inputpars[indices]
real_spec = spec[indices]


# reNorm the parameters 
inputpars_sampled= denorm_pars(inputpars_sampled,mins=np.array([1.,0.5,5,0.,3]),maxs=np.array([3.4,10,1000,4.7,87]))

# Energy bins
epi = np.load('Ebins.npy')

for i,specs in enumerate(outputs_sampled):
    specs = denorm_spec(specs)
    pars = inputpars_sampled[i]
    # There has some normalization problem if one use the load_xspec_model to generate the spectrum
    xspec_spec =denorm_spec(real_spec[i])
    neural_spec = specs
    pars =  np.around(pars, 2)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epi, neural_spec, label='Neural')
    plt.plot(epi, xspec_spec, label='Xspec')
    plt.title(f'Random test $\Gamma$ {pars[0]}, Afe {pars[1]}, Ecut {pars[2]}, logxi {pars[3]}, Incl {pars[4]}')
    plt.xlabel('E [keV]')
    plt.ylabel('EF_E')
    plt.yscale('log') 
    plt.xscale('log') 
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epi, neural_spec / xspec_spec, label='Ratio [NN/XSPEC]')
    plt.title(f'Ratio at  $\Gamma$ {pars[0]}, Afe {pars[1]}, Ecut {pars[2]}, logxi {pars[3]}, Incl {pars[4]}')
    plt.xlabel('E [keV]')
    plt.ylabel('Ratio')
    # plt.yscale('log' )
    plt.xscale('log')
    plt.legend()

    plt.tight_layout()
    path = f"./predict_plot/{i}.png"
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(f"./predict_plot/{i}.png",dpi=300)
    plt.close()






                  

