import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd 
import os

# plot default
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['text.usetex'] = True   
matplotlib.rcParams['figure.dpi'] = 200

# energy bins
epi = np.load('Ebins.npy')

# check the path exists
if not os.path.exists('./TEM_PLOT'):
    os.mkdir('TEM_PLOT')

def denorm_spec(x):
    """
        denorm the spectrum 
        x : spectrum
    """
    x = x+1 
    return np.exp(x)
def check_zero(x):
    if np.any(x == 0) or np.any(x < 0):
        return False
    else:
        return True
def denorm_pars(normalized_pars, mins = np.array([1,0.5,5,0,3]),maxs=np.array([3.4,10,1000,4.7,87])):
    """
        renorm the pars 
        normalized_pars: parameters
    """
    # Reshape mins and maxs to ensure they are broadcasted correctly with normalized_pars
    mins = mins.reshape(1, -1)
    maxs = maxs.reshape(1, -1)
    # Perform the denormalization
    original_pars = normalized_pars * (maxs - mins) + mins
    return original_pars


def plot_spectrum(fake,real,pars,title,mins,maxs):

    """
    Plot the spectrum when training
    Args:
        fake: neural network spectrum
        real: xspec spectrum
        pars: corrosponding parameters
        title: image title
        mins: minimum value of parameters
        maxs: maximum value of parameters
    """

    # denorm the parameters
    pars= denorm_pars(pars,mins,maxs)
    # reduce to .2e 
    pars =  np.around(pars, 2)
    # initialize the image
    image = None
    # denorm the spectrum
    fake = denorm_spec(fake)
    real = denorm_spec(real)

    # plot
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    if check_zero(fake):
        plt.plot(epi, fake, label='fake')
    plt.plot(epi, real, label='real')
    plt.title(f'{title} - Fake vs Real at {pars}')
    plt.xlabel('E [keV]')
    plt.ylabel('EF_E')
    plt.yscale('log') 
    plt.xscale('log') 
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epi, fake / real, label='fake/real')
    plt.title(f'Ratio at {pars}')
    plt.xlabel('E [keV]')
    plt.ylabel('Ratio')
    plt.yscale('log' )
    plt.xscale('log')
    plt.legend()

    plt.tight_layout()
    path = f"./TEM_PLOT/{title}.png"
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(f"./TEM_PLOT/{title}.png")
    plt.close()
    image = plt.imread(f"./TEM_PLOT/{title}.png")
    return image


    