import torch
import torch.nn as nn
import numpy as np
import argparse
import load_data
from Plot import plot_spectrum
import time
from torchinfo import summary
import model
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=5, help="number of classes for dataset")
parser.add_argument("--sp_size", type=int, default=1000, help="size of each spectra length")
parser.add_argument("--par_size", type=int, default=5, help="size of each parameters length")
parser.add_argument("--epoch_interval", type=int, default=10, help="iteration times")
parser.add_argument("--sample_interval", type=int, default=10, help="sample_interval for test")
parser.add_argument("--rat", type=int, default=1, help="weight of physical loss")
opt =parser.parse_args()

# energy bins
epi = np.load('Ebins.npy')

# find where the value include most emission line (0.2-1keV)
mask = (epi >= 0.2) & (epi <= 1)
indices = np.array(np.where(mask))[0]

def physical_loss(real_spectrum, predicted_spectrum):
    """
    Define the physical loss function focus on the emission line part

    Args:
        real_spectrum: real spectrum from xspec
        predicted_spectrum: spectrum from neura network

    """
    up_boundary = indices[0]
    bo_boundary = indices[-1]
    # Get the slice of index range
    linerang_real = real_spectrum[:, up_boundary:bo_boundary]
    linerang_pred = predicted_spectrum[:,up_boundary:bo_boundary]
    
    # Get the summation
    realsum = torch.sum(linerang_real,dim=-1)
    presum = torch.sum(linerang_pred,dim=-1)

    # Do MSE for this part
    emis_dif = torch.mean(torch.abs(realsum-presum)**2/linerang_pred.size(1))

    return emis_dif 

# Get gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU is available")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("No GPU available")

# Tensorboard    
writer = SummaryWriter(log_dir="runs_inic/training_logs")

# model to gpu
generator = model.Generator().to(device)

# model to float64
generator = generator.double()

# summary 
batch_size = 32
summary(generator, input_size=(batch_size, opt.par_size), col_names = ["input_size", "output_size", "num_params"])

# load data
train_loader,test_loader= load_data.load_data(path ='/data/home/yimin2/advanced_model/DataSet/DataSet100000_paper.h5',
                                   batchsize=opt.batch_size,
                                   device=device,
                                   mins = np.array([1.,0.5,5,0.,3]),
                                   maxs = np.array([3.4,10,1000,4.7,87]),
                                   normalize='log',
                                   datatype ='DataLoader')

# Define MAE loss function
criterion = nn.L1Loss() 

# Adam optimizer 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)

# Variable to store the loss
lossPlot = np.zeros(opt.n_epochs)

# Training
for epoch in range(opt.n_epochs):
    # get time
    epoch_time = time.time() 
    # total loss
    epoch_loss = 0.
    # training module
    generator.train()
    for i,(pars,spec) in enumerate(train_loader):

        # make sure data has move 
        pars = pars.double().to(device)
        spec = spec.double().to(device)

        # initialize gradient
        optimizer_G.zero_grad()

        # generate spectrum
        gen_spec = generator(pars)

        # loss function
        g_loss = criterion(gen_spec, spec)+opt.rat*physical_loss(gen_spec,spec)

        # back propagation
        g_loss.backward()

        # update gradient
        optimizer_G.step()

        # get total loss for this batch
        epoch_loss += g_loss.item()

    lossPlot[epoch]=epoch_loss
    print(
        "[Epoch %d/%d] [G loss: %f]"
        % (epoch, opt.n_epochs, epoch_loss)
    )

    # Evalution
    if epoch%opt.epoch_interval == 0:
        generator.eval()
        with torch.no_grad():
            for j,(pars,spec) in enumerate(test_loader):
                if j%opt.sample_interval ==0:
                    spec_output = generator(pars)
                    spectrum_image = plot_spectrum(spec_output[0].cpu().detach().numpy(),
                                           spec[0].cpu().detach().numpy(),
                                           pars[0].cpu().detach().numpy(),
                                           title=f"EPOCH{epoch}_{j}",
                                           mins = np.array([1.,0.5,5,0.,3]),
                                           maxs =  np.array([3.4,10,1000,4.7,87]))
                    writer.add_image(f"{pars}/epoch", spectrum_image, epoch, dataformats='HWC')
    writer.add_scalar('Loss/LOSS', epoch_loss/opt.batch_size, epoch)
torch.save(generator.state_dict(), f'model_dict.pth')
torch.save(generator,f'model.pth')
writer.close()
