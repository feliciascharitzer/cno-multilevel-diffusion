import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import itertools
import numpy as np
import torch
import torchvision
from torchvision import datasets
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from unet_no_att import UNet
from fno import *
from cno import *
from metrics import *
from unet_no_att import *
from sde import * 
from priors import *
from utils import get_samples, makedirs, get_logger, get_samples_batched, epsTest
import os
import datetime
from torch.utils.data import Dataset
import csv

#--------------------------------------

# Adapted from https://arxiv.org/abs/2303.04772v3  
 
#--------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def training(seed, model, args, out_file=None):
    """
    training the score function

    :param seed: random seed, ensure reproducibility within the same seed
    :param model: function approximator for the score function
    :param args: hyperparameters
    :return:
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    pool = torch.nn.AvgPool2d(2)
    #define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(2*args.input_height)])
    #download and load the training data
    if args.dataset == 'MNIST':
        trainset = datasets.MNIST(root='~/.pytorch/data', train=True, download=False, transform=transform)
    elif args.dataset == 'FashionMNIST':
        trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    else:
        raise argparse.ArgumentTypeError(f"Not a predefined dataset")

    #train on 1/10 of dataset to test implementation
    if args.subset:
        subset_indices = np.random.choice(len(trainset), 6000, replace=False)
        train_subset = torch.utils.data.Subset(trainset, subset_indices)
        train_set, val_set = torch.utils.data.random_split(train_subset, [6000-args.num_samples_mmd, args.num_samples_mmd])  
    else:
        train_set, val_set = torch.utils.data.random_split(trainset, [60000-args.num_samples_mmd, args.num_samples_mmd])
    
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.num_samples_mmd,
                                          shuffle=True, num_workers=4)
    val_samp = next(iter(valloader))[0]

    val_samp_pool = pool(val_samp).view(args.num_samples_mmd,args.input_height**2).to('cpu')
    val_samp = val_samp.view(args.num_samples_mmd,(2*args.input_height)**2).to('cpu')

    logger.info('NUMBER OF PARAMETERS:')
    logger.info(sum(p.numel() for p in model.parameters()))

    fwd_sde = VariancePreservingSDE(args.prior, alpha_min=0.1, alpha_max=20.0, T=1. )
    rev_sde = PluginReverseSDE(args.prior, fwd_sde, model, 1., vtype='Rademacher', debias=False).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("-------------------------")
    logger.info(str(optim)) #optimizer info
    logger.info("-------------------------\n")

    logger.info("The program is running on a %s.", 'GPU' if torch.cuda.is_available() else 'CPU')
    
    min_mmd = 1000
    min_mmd_epoch = 0.
    
    history = []
    loss_list = []
    mmd_list = []
    mmd2_list = []

    rev_sde.train()
    for ep in range(args.n_epochs):
        if ep % 10 == 0:
            logger.info("epoch: %s", str(ep))
        avg_loss = 0.0
        for k,(x,y) in enumerate(trainloader):
            x = x.to(device) 
            loss = rev_sde.dsm(pool(x)).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()*x.shape[0]

        avg_loss /= len(trainset)


        if ep % args.val_freq ==0:

            with torch.no_grad():
                y0 = get_samples_batched(rev_sde, 1, args.input_height, args.input_height, args.num_steps, args.num_samples_mmd).view(args.num_samples_mmd,(args.input_height)**2)
                y02 = get_samples_batched(rev_sde, 1, 2*args.input_height, args.input_height, args.num_steps, args.num_samples_mmd).view(args.num_samples_mmd,(2*args.input_height)**2)
            
            y0 = torch.clamp(y0,0.,1.).to('cpu')
            y02 = torch.clamp(y02,0.,1.).to('cpu')
            
            riesz_mmd = mmd(y0, val_samp_pool)
            riesz2_mmd = mmd(y02, val_samp)
            
            history.append([ep,avg_loss,riesz_mmd,riesz2_mmd]) 
            loss_list.append(avg_loss)
            mmd_list.append(riesz_mmd)
            mmd2_list.append(riesz2_mmd)

            print(riesz_mmd)
            logger.info('epoch:%05d\t loss:%1.2e \t mmd:%1.2e \t mmdhigh:%1.2e ' % (ep, avg_loss, riesz_mmd, riesz2_mmd))

            if riesz2_mmd < min_mmd:
                torch.save(rev_sde, ("%s-min_checkpoint.pt") % (out_file))
                min_mmd = riesz2_mmd
                min_mmd_epoch = ep

        #visualization
        if ep % args.viz_freq==0:
            y0 = get_samples(rev_sde, 1, args.input_height, args.input_height, args.num_steps, args.num_samples)[0]
            y0 = torch.clamp(y0,0.,1.)

            plt.figure()
            image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))

            plt.imshow(image_grid.cpu().numpy(), cmap='gray')
            plt.savefig(("%s-mnist_samples_28ML-%d.png") % (out_file, ep + 1))
            plt.close()
            y0 = get_samples(rev_sde, 1, 2*args.input_height, args.input_height, args.num_steps, args.num_samples)[0]

            y0 = torch.clamp(y0,0.,1.)

            plt.figure()
            image_grid = torchvision.utils.make_grid(y0, nrow=4, padding=5).permute((1, 2, 0))

            plt.imshow(image_grid.cpu().numpy(), cmap='gray')
            plt.savefig(("%s-mnist_samples_56ML-%d.png") % (out_file, ep + 1))
            plt.close()

    logger.info('MINIMUM EPOCH AND MMD')
    logger.info(min_mmd_epoch)
    logger.info(min_mmd)

    plot_file = f"{args.model}_{args.prior}_modes{args.modes}_{args.plot_run}.csv"

    with open(plot_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["seed", "epoch", "loss", "mmd_lowres", "mmd_highres"])
        for epoch, (loss, mmd_low, mmd_high) in enumerate(zip(loss_list, mmd_list, mmd2_list)):
            writer.writerow([args.seed, epoch, loss, mmd_low, mmd_high])

    return rev_sde, history, loss_list, mmd_list

def choose_prior(string):
    if string.lower() == "fno":
        return FNOprior(k1=28,k2=args.modes)

    elif string.lower() == "standard":
        return StandardNormal()

    elif string.lower() == "lap_conv":
        K = torch.zeros(3,3)
        hx = 1.0/args.input_height
        hy = 1.0/args.input_height
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)
        return ImplicitConv(K,scale=10)

    elif string.lower() == "combined_conv":
        K = torch.zeros(3,3)
        hx = 1.0/args.input_height
        hy = 1.0/args.input_height
        K[1,1] = 2.0/(hx**2) + 2.0/(hy**2)
        K[0,1] = -1.0/(hy**2)
        K[1,0] = -1.0/(hx**2)
        K[2,1] = -1.0/(hy**2)
        K[1,2] = -1.0/(hx**2)
        return CombinedConv(K,k1=args.modes,k2=args.modes,scale=10)

    else:
        raise argparse.ArgumentTypeError(f"Invalid class")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')

    lrvalues = [1e-3, 1e-4]
    batches = [256]
    numstep_values = [200]
    priorchoices = ['fno','combined_conv','lap_conv','standard']
    modelchoices = ['unet','fno','cno']
    modeschoices = [8, 12, 14, 15]
    widthchoices = [32, 64, 128]

    combinations = list(itertools.product(lrvalues, batches, numstep_values, priorchoices, modelchoices, modeschoices,widthchoices))
    parser.add_argument('--dataset', type=str, default='MNIST', choices = ['MNIST', 'FashionMNIST'], help='dataset types')
    parser.add_argument('--n_epochs', type=int, default=501, help='ADAM epoch')
    parser.add_argument('--lr', type=float,default=1e-4, choices = lrvalues, help='ADAM learning rate')
    parser.add_argument('--batch_size', type=int, default=256, choices = batches, help='number of training samples in each batch')
    parser.add_argument('--num_samples', type=int, default=16, help='number of samples for visualization')
    parser.add_argument('--num_samples_mmd', type=int, default=10000, help='number of samples for validation')
    parser.add_argument('--num_steps', type=int, default=200, choices = numstep_values, help='number of SDE timesteps')
    parser.add_argument('--input_height', type=int, default=28, help='starting image dimensions')
    parser.add_argument('--prior_name', type=str, default='combined_conv', choices = priorchoices, help="prior setup")
    parser.add_argument('--width', type=int, default=64, choices = widthchoices, help="prior setup")
    parser.add_argument('--model', type=str, default='fno', choices = modelchoices, help='nn model')
    parser.add_argument('--modes', type=int, default=14, choices = modeschoices, help='cutoff modes in FNO')
    parser.add_argument('--viz_freq', type=int, default=10, help='how often to store generated images')
    parser.add_argument('--val_freq', type=int, default=10, help='validation freq')
    parser.add_argument('--seed', type=int, default=1, help='seed for random number generator')
    parser.add_argument('--out_dir', type=str, default='result', help='directory for result')
    parser.add_argument('--out_file', type=str, default='result', help='base file name for result')
    parser.add_argument('--subset', type=int, default=0, help='indicate 1 for training on 1/10th of subset')
    parser.add_argument('--plot_run', type=int, default=0, help='indicate whether the run is part of a plot cycle')

    args = parser.parse_args()
    
    #testing on smaller dataset affects number of validation samples
    if args.subset:
        args.num_samples_mmd = 1000
    
    args.prior = choose_prior(args.prior_name)
    if args.model == "fno":
        model = FNO2d(args.modes,args.modes,args.width).to(device) 
    
    if args.model == "cno":
        model = CNO(in_dim=1, in_size=args.input_height, N_layers=3, N_res=4, channel_multiplier=64,
                    latent_lift_proj_dim=32, time_steps=args.num_steps)

    else:
        model = UNet(
        input_channels=1,
        input_height=args.input_height,
        ch=64,
        ch_mult=(1, 2, 2),
        num_res_blocks=4,
        resamp_with_conv=True,).to(device)

    input_channels = 1
    
    start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    if args.out_file is not None:
        out_file = os.path.join(args.out_dir, '{:}-{:}_model_{:}_prior_{:}'.format(start_time,args.out_file,args.model, args.prior))
    else:
        out_file=None
    
    makedirs(args.out_dir)
    logger = get_logger(logpath= out_file + '.txt', filepath=os.path.abspath(__file__))
    
    print(f"Chosen prior: {args.prior}")
    print(f"Repr of chosen prior: {repr(args.prior)}")
    for arg, value in vars(args).items():
        logger.info("%s: %s", arg, value)
    
    rev_sde, history, loss_list, mmd_list = training(args.seed, model, args, out_file=out_file)
