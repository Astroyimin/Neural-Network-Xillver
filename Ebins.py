import xspec
import pandas as pd
import numpy as np

"""
    Generate the energy bins
"""
xspec.AllModels.lmod('relxill','/data/models/relxill_v2.3')
xspec.AllModels.setEnergies(f"0.0717702 1000.1 3000 log")
model = xspec.Model('xillver')
xspec.Plot.device = '/null'
xspec.Plot.add = True
xspec.Plot.addCommand(f'wd Ebins.dat')
xspec.Plot('eemodel') 
xspec.Plot.addCommand(f'exit')
xspec.Plot.commands = ()
xspec.AllModels.clear()
df = pd.read_csv('Ebins.dat', sep='\s+', header=None, skiprows=3)
ebins = np.array(df[0])
np.save('Ebins.npy', ebins)