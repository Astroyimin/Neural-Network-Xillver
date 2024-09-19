import os
import xspec
import numpy as np
import pandas as pd
import itertools
import h5py
import concurrent.futures

# load necessary model 'relxill'
xspec.AllModels.lmod('relxill','/data/models/relxill_v2.3')
# model name
Name = 'xillver'
skipnum = 0 
def load_model (n,pars,name,father,resolution):
    """
    Args:
        n: index of pars 
        pars: xillver parameters
        resolution: Energy bins
        father: save path
    """
    dir = os.path.join(father,f'source_test{resolution}bins')
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir,str(n)+'.dat')
    if os.path.exists(path):
        return path
    xspec.AllModels.setEnergies(f"0.0717702 1000.1 {resolution} log")
    xspec.AllModels.clear()
    model = xspec.Model(name)
    model.xillver.gamma.values = pars[0]
    model.xillver.Afe.values =pars[1]
    model.xillver.Ecut.values = pars[2]
    model.xillver.logxi.values = pars[3]
    model.xillver.Incl.values = pars[4]
    xspec.AllModels.show()
    xspec.Plot.device = '/null'
    xspec.Plot.add = True
    xspec.Plot.addCommand(f'wd {path}')
    xspec.Plot('eemodel') 
    xspec.Plot.addCommand(f'exit')
    xspec.Plot.commands = ()
    xspec.AllModels.clear()
    return path

def get_data(path):
    df = pd.read_csv(path, sep='\s+', header=None, skiprows=3)
    spec = np.array(df[2])
    return spec
def get_dataSet(parsSets,Name,sample_size,father,resolution):

    """
    Args :
        parsSets: Sampled parameter sets
        Name : name of model
        i: index of parameter group
        sample_size: sample number
        father: save path
        resolution: Energy bins of spectra
    """
    sampled_data = parsSets[:sample_size, :]
    sampled_spec = np.zeros((sample_size, resolution))
    skipnum = 0
    for n,parset in enumerate(sampled_data):
        path = load_model(n,parset,Name,father,resolution)
        spec = get_data(path)
        if len(spec)!=resolution:
            skipnum =+ 1 
            continue
        else: 
            sampled_spec[n] = get_data(path)
        # os.remove(path)
    return  sampled_spec,sampled_data,skipnum

# parameters boundary
boundarys = [(1,3.3),(0.5,9.9),(5,995),(1.0,4.6),(3,84)]
# parameters change step
steps = [0.1,0.1,5,0.1,3]
# total range 
ranges = [np.arange(start, stop+step, step) for (start, stop), step in zip(boundarys, steps)]
parsSets = [tuple(combination) for combination in itertools.product(*ranges)]
# Transport to np array
parsSets = np.array(parsSets)
# Get spectrum from xspec
sample_size = 100000
spec,pars,skipnum= get_dataSet(parsSets,Name,sample_size,'DataSet',resolution=2999)
with h5py.File(f'./DataSet/DataSet{sample_size}_paper.h5', 'w') as hf:
    hf.create_dataset(f'spec', data=spec)
    hf.create_dataset(f'pars', data=pars)
    print(' Total num of mis-match dimension :',skipnum)
