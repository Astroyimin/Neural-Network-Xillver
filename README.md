# Neural Network Xillver
 This repository is to convert xillver table model to neural netwok. The goal is to accelerate the reading table process in relxill 


## Model Structure

The model has 6 linear layers, after that a `tanh` activation is applied.

**Physical loss function**

Due to many line emission features at 0.1-1keV, we introduce a physical reasonable loss function, making the training pay more attention on this part.

```
 linerang_real = real_spectrum[:, up_boundary:bo_boundary]
 linerang_pred = predicted_spectrum[:,up_boundary:bo_boundary]
 
 # Get the summation
 realsum = torch.sum(linerang_real,dim=-1)
 presum = torch.sum(linerang_pred,dim=-1)

 # Do MSE for this part
 emis_dif = torch.mean(torch.abs(realsum-presum)**2/linerang_pred.size(1))
```

## Data preprocess

The model receive 5 physical parameter as describe in [Garcia 2013](https://arxiv.org/abs/1303.2112). 

1. Photon Index
2. Iron Abundance
3. High energy Ecut [keV]
4. Logxi
5. Inclination

The boundary is same with corresponding table model. If one want to use this model to get reflected spectrum, **the normalization of input parameters is necessary**. Our normalization are showed in `load_data.py`, the normalization of parameters as follows:

```
mins = mins.reshape(1,-1)
maxs = maxs.reshape(1,-1)
pars = (pars-mins)/(maxs-mins)
```

Moreover, the training spectrum is normalized as follows:

```
spec = np.log(spec)
spec = (spec -spec.min())/(spec.max()-spec.min())-1
```

## Spectra properties

As same with *xillver* [see [Garcia 2013](https://arxiv.org/abs/1303.2112)], the spectrum has 3000 energy grids, the fits file is available on [relxill](https://www.sternwarte.uni-erlangen.de/~dauser/research/relxill/). 

Our neural network model shows good ability to map the input parameters to the spectrum. The result is show in `result`. 

Trained model store in `model`

# Acknowledge
[relxill v2.3.3](https://www.sternwarte.uni-erlangen.de/~dauser/research/relxill/),[HEAsoft v6.31.1](https://heasarc.gsfc.nasa.gov/docs/software/heasoft/),Python 3.8.19,cuda 12.1,torch 2.4.0+cu121




