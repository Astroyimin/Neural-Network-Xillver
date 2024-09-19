# Neural Network Xillver
 This repository is to convert xillver table model to neural netwok. The goal is to accelerate the reading table process in relxill 

**Model structure**
The model has 6 linear layers, follow a tanh activation.

Due to many line emission features at 0.1-1keV, we introduce a physical reasonable loss function, which correspond the related part in spectrum.

```
 linerang_real = real_spectrum[:, up_boundary:bo_boundary]
 linerang_pred = predicted_spectrum[:,up_boundary:bo_boundary]
 
 # Get the summation
 realsum = torch.sum(linerang_real,dim=-1)
 presum = torch.sum(linerang_pred,dim=-1)

 # Do MSE for this part
 emis_dif = torch.mean(torch.abs(realsum-presum)**2/linerang_pred.size(1))

 return emis_dif 
```

**Spectra properties**
As same with *xillver* [see [Garcia 2013](https://arxiv.org/abs/1303.2112)], the spectrum has 3000 energy grids, the fits file is available on [relxill](https://www.sternwarte.uni-erlangen.de/~dauser/research/relxill/).



