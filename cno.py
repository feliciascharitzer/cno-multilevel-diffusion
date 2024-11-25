import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from cno_training.filtered_networks import LReLu, LReLu_regular, LReLu_torch #Either "filtered LReLU" or regular LReLu
from cno_debug_tools import format_tensor_size
#------
import math
from torch.nn.init import _calculate_fan_in_and_fan_out

#--------------------------------------

# CNO implementation adapted from https://arxiv.org/abs/2302.01178 (CNO2d_original_version)
# Including time conditioning from https://arxiv.org/abs/2405.19101
 
#--------------------------------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#--------------------------------------
# FiLM: Visual Reasoning with a General Conditioning Layer
# See https://arxiv.org/abs/1709.07871

class FILM(torch.nn.Module):
    def __init__(self,
                 channels,
                 dim = [0,2,3],
                 s = 128,
                 intermediate = 128):
        super(FILM, self).__init__()
        self.channels = channels
        self.s = s

        self.inp2lat_sacale = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2scale = nn.Linear(in_features=intermediate, out_features=channels)

        self.inp2lat_bias = nn.Linear(in_features=1, out_features=intermediate,bias=True)
        self.lat2bias = nn.Linear(in_features=intermediate, out_features=channels)

        self.inp2lat_sacale.weight.data.fill_(0)
        self.lat2scale.weight.data.fill_(0)
        self.lat2scale.bias.data.fill_(1)

        self.inp2lat_bias.weight.data.fill_(0)
        self.lat2bias.weight.data.fill_(0)
        self.lat2bias.bias.data.fill_(0)

        if dim == [0,2,3]:
            self.norm = nn.BatchNorm2d(channels)
        elif dim == [2,3]:
            self.norm = nn.InstanceNorm2d(channels, affine=True)
        elif dim == [1,2,3]:
            self.norm = nn.LayerNorm([channels, s, s])
        else:
            self.norm = nn.Identity()

    def forward(self, x, timestep):

        x = self.norm(x)
        timestep = timestep.reshape(-1,1).type_as(x)
        scale     = self.lat2scale(self.inp2lat_sacale(timestep))
        bias      = self.lat2bias(self.inp2lat_bias(timestep))
        scale = scale.unsqueeze(2).unsqueeze(3)
        scale     = scale.expand_as(x)
        bias  = bias.unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scale + bias

#-----------------------------------------

class CNOBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu',

                 is_time = 4,
                 nl_dim = [0],
                 time_steps = 5,
                 lead_time_features = 512
                 ):
        super(CNOBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels #important for time conditioning
        self.in_size  = in_size
        self.out_size = out_size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        self.nl_dim = nl_dim

        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.in_cutoff  = self.in_size / cutoff_den
        self.out_cutoff = self.out_size / cutoff_den

        self.in_halfwidth =  half_width_mult*self.in_size - self.in_size / cutoff_den
        self.out_halfwidth = half_width_mult*self.out_size - self.out_size / cutoff_den

        #-----------------------------------------

        # We apply Conv -> BN (optional) -> Activation
        # Up/Downsampling happens inside Activation

        pad = (self.conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = self.in_channels, out_channels=self.out_channels,
                                           kernel_size=self.conv_kernel,
                                           padding = pad)

        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(self.out_channels)
        else:
            self.batch_norm = nn.Identity()

        if activation == "cno_lrelu":
            self.activation  = LReLu(in_channels           = self.out_channels, #In _channels is not used in these settings
                                     out_channels          = self.out_channels,
                                     in_size               = self.in_size,
                                     out_size              = self.out_size,
                                     in_sampling_rate      = self.in_size,
                                     out_sampling_rate     = self.out_size,
                                     in_cutoff             = self.in_cutoff,
                                     out_cutoff            = self.out_cutoff,
                                     in_half_width         = self.in_halfwidth,
                                     out_half_width        = self.out_halfwidth,
                                     filter_size           = filter_size,
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.out_channels, #In _channels is not used in these settings
                                          out_channels          = self.out_channels,
                                          in_size               = self.in_size,
                                          out_size              = self.out_size,
                                          in_sampling_rate      = self.in_size,
                                          out_sampling_rate     = self.out_size)
        elif activation == "lrelu":
            self.activation  = LReLu_regular(in_channels           = self.out_channels, #In _channels is not used in these settings
                                             out_channels          = self.out_channels,
                                             in_size               = self.in_size,
                                             out_size              = self.out_size,
                                             in_sampling_rate      = self.in_size,
                                             out_sampling_rate     = self.out_size)
        else:
            raise ValueError("Please specify different activation function")

        #time conditioning
        self.is_time = is_time
        self.time_steps = time_steps

        if is_time == 1 or is_time == True:
            self.time_steps = time_steps
            self.in_norm_conditioner = FILM(out_channels,
                                            dim = nl_dim,
                                            s = self.in_size)
            self.batch_norm = nn.Identity()

    def forward(self, x, time, scaling=1.0):
        
        x = self.convolution(x)
        x = self.batch_norm(x)
        if self.is_time == 1 or self.is_time == True:
            x = self.in_norm_conditioner(x, time)
        x = self.activation(x, scaling)
        return x

#------------------------------------------------------------------------------

# Contains CNOBlock -> Convolution -> BN

class LiftProjectBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_size,
                 out_size,
                 latent_dim = 64,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu',

                 is_time = True,
                 time_steps = 5,
                 lead_time_features = 512
                 ):
        super(LiftProjectBlock, self).__init__()

        #important for time conditioning
        self.out_channels = out_channels

        self.inter_CNOBlock = CNOBlock(in_channels = in_channels,
                                       out_channels = latent_dim,
                                       in_size = in_size,
                                       out_size = out_size,
                                       cutoff_den = cutoff_den,
                                       conv_kernel = conv_kernel,
                                       filter_size = filter_size,
                                       lrelu_upsampling = lrelu_upsampling,
                                       half_width_mult  = half_width_mult,
                                       radial = radial,
                                       batch_norm = batch_norm,
                                       activation = activation,
                                       is_time = is_time,
                                       time_steps = time_steps,
                                       lead_time_features = lead_time_features)

        pad = (conv_kernel-1)//2
        self.convolution = torch.nn.Conv2d(in_channels = latent_dim, out_channels=out_channels,
                                           kernel_size=conv_kernel, stride = 1,
                                           padding = pad)

    def forward(self, x, time, scaling=1.0):
        x = self.inter_CNOBlock(x, time, scaling)
        x = self.convolution(x)
        return x

#------------------------------------------------------------------------------

# Residual Block containts:
# Convolution -> BN -> Activation -> Convolution -> BN -> SKIP CONNECTION

class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 size,
                 cutoff_den = 2.0001,
                 conv_kernel = 3,
                 filter_size = 6,
                 lrelu_upsampling = 2,
                 half_width_mult  = 0.8,
                 radial = False,
                 batch_norm = True,
                 activation = 'cno_lrelu',

                 is_time = 4,
                 nl_dim = [0],
                 time_steps = 5,
                 lead_time_features = 512
                 ):
        super(ResidualBlock, self).__init__()

        self.channels = channels #important for time conditioning
        self.size  = size
        self.conv_kernel = conv_kernel
        self.batch_norm = batch_norm
        self.nl_dim = nl_dim

        #---------- Filter properties -----------
        self.citically_sampled = False #We use w_c = s/2.0001 --> NOT critically sampled

        if cutoff_den == 2.0:
            self.citically_sampled = True
        self.cutoff  = self.size / cutoff_den
        self.halfwidth =  half_width_mult*self.size - self.size / cutoff_den

        #-----------------------------------------

        pad = (self.conv_kernel-1)//2
        self.convolution1 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels,
                                            kernel_size=self.conv_kernel, stride = 1,
                                            padding = pad)
        self.convolution2 = torch.nn.Conv2d(in_channels = self.channels, out_channels=self.channels,
                                            kernel_size=self.conv_kernel, stride = 1,
                                            padding = pad)

        if self.batch_norm:
            self.batch_norm1  = nn.BatchNorm2d(self.channels)
            self.batch_norm2  = nn.BatchNorm2d(self.channels)
        else: #in this implementation we always apply batch norm
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        if activation == "cno_lrelu":
            self.activation  = LReLu(in_channels           = self.channels, #In _channels is not used in these settings
                                     out_channels          = self.channels,
                                     in_size               = self.size,
                                     out_size              = self.size,
                                     in_sampling_rate      = self.size,
                                     out_sampling_rate     = self.size,
                                     in_cutoff             = self.cutoff,
                                     out_cutoff            = self.cutoff,
                                     in_half_width         = self.halfwidth,
                                     out_half_width        = self.halfwidth,
                                     filter_size           = filter_size,
                                     lrelu_upsampling      = lrelu_upsampling,
                                     is_critically_sampled = self.citically_sampled,
                                     use_radial_filters    = False)
        elif activation == "cno_lrelu_torch":
            self.activation = LReLu_torch(in_channels           = self.channels, #In _channels is not used in these settings
                                          out_channels          = self.channels,
                                          in_size               = self.size,
                                          out_size              = self.size,
                                          in_sampling_rate      = self.size,
                                          out_sampling_rate     = self.size)
        elif activation == "lrelu":

            self.activation = LReLu_regular(in_channels           = self.channels, #In _channels is not used in these settings
                                            out_channels          = self.channels,
                                            in_size               = self.size,
                                            out_size              = self.size,
                                            in_sampling_rate      = self.size,
                                            out_sampling_rate     = self.size)
        else:
            raise ValueError("Please specify different activation function")

        #time conditioning
        self.is_time = is_time
        if self.is_time==1 or self.is_time == True:
            self.time_steps = time_steps
            self.in_norm_conditioner1 = FILM(channels,
                                            dim = nl_dim,
                                            s = self.size)
            self.in_norm_conditioner2 = FILM(channels,
                                            dim = nl_dim,
                                            s = self.size)
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

    def forward(self, x, time, scaling=1.0):
        out = self.convolution1(x)
        out = self.batch_norm1(out)

        if self.is_time == 1 or self.is_time == True:
            out = self.in_norm_conditioner1(out, time)
        out = self.activation(out, scaling)
        out = self.convolution2(out)

        out = self.batch_norm2(out)
        if self.is_time == 1 or self.is_time == True:
            out = self.in_norm_conditioner2(out, time)

        x = x + out
        del out

        return x

#------------------------------------------------------------------------------

#CNO NETWORK:
class CNO(nn.Module):
    def __init__(self,
                 in_dim,                    # Number of input channels.
                 in_size,                   # Input spatial size
                 N_layers,                  # Number of (D) or (U) blocks in the network
                 N_res = 1,                 # Number of (R) blocks per level (except the neck)
                 N_res_neck = 6,            # Number of (R) blocks in the neck
                 channel_multiplier = 32,   # How the number of channels evolve?
                 conv_kernel=3,             # Size of all the kernels
                 cutoff_den = 2.0001,       # Filter property 1.
                 filter_size=6,             # Filter property 2.
                 lrelu_upsampling = 2,      # Filter property 3.
                 half_width_mult  = 0.8,    # Filter property 4.
                 radial = False,            # Filter property 5. Is filter radial?
                 batch_norm = True,         # Add BN? We do not add BN in lifting/projection layer
                 out_dim = 1,               # Target dimension
                 out_size = 1,              # If out_size is 1, Then out_size = in_size. Else must be int
                 expand_input = False,      # Start with original in_size, or expand it (pad zeros in the spectrum)
                 latent_lift_proj_dim = 64, # Intermediate latent dimension in the lifting/projection layer
                 add_inv = True,            # Add invariant block (I) after the intermediate connections?
                 activation = 'cno_lrelu',  # Activation function can be 'cno_lrelu' or 'lrelu'

                 time_steps = 5,
                 is_time = 1,
                 nl_dim = [1] #needed for FILM, gives dimension of norm calculation
                 ):

        super(CNO, self).__init__()

        ###################### Define the parameters & specifications #################################################


        # Number od (D) & (U) Blocks
        self.N_layers = int(N_layers)

        # Input is lifted to the half on channel_multiplier dimension
        self.lift_dim = channel_multiplier//2
        self.in_dim = in_dim
        self.out_dim  = out_dim

        #Should we add invariant layers in the decoder?
        self.add_inv = add_inv

        # The growth of the channels : d_e parametee
        self.channel_multiplier = channel_multiplier

        # Is the filter radial? We always use NOT radial
        if radial ==0:
            self.radial = False
        else:
            self.radial = True

        ###################### Define evolution of the number features ################################################

        # How the features in Encoder evolve (number of features)
        self.encoder_features = [self.lift_dim]
        for i in range(self.N_layers):
            self.encoder_features.append(2 ** i *   self.channel_multiplier)

        # How the features in Decoder evolve (number of features)
        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.N_layers):
            self.decoder_features_in[i] = 2*self.decoder_features_in[i] #Pad the outputs of the resnets

        self.inv_features = self.decoder_features_in
        self.inv_features.append(self.encoder_features[0] + self.decoder_features_out[-1])

        _lead_time_features = max(self.inv_features)
        if self.encoder_features[-1]>_lead_time_features:
            _lead_time_features = self.encoder_features[-1]
        _lead_time_features = 2*_lead_time_features
        ################

        ###################### Define evolution of sampling rates #####################################################

        if not expand_input:
            latent_size = in_size # No change in in_size
        else:
            down_exponent = 2 ** N_layers
            latent_size = in_size - (in_size % down_exponent) + down_exponent # Jump from 64 to 72, for example

        #Are inputs and outputs of the same size? If not, how should the size of the decoder evolve?
        if out_size == 1:
            latent_size_out = latent_size
        else:
            if not expand_input:
                latent_size_out = out_size # No change in in_size
            else:
                down_exponent = 2 ** N_layers
                latent_size_out = out_size - (out_size % down_exponent) + down_exponent # Jump from 64 to 72, for example

        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.N_layers + 1):
            self.encoder_sizes.append(latent_size // 2 ** i)
            self.decoder_sizes.append(latent_size_out // 2 ** (self.N_layers - i))

        print(f"****** encoder_sizes: {self.encoder_sizes}, decoder_sizes: {self.decoder_sizes}")

        ###################### Define Projection & Lift ##############################################################

        self.lift = LiftProjectBlock(in_channels  = in_dim,
                                     out_channels = self.encoder_features[0],
                                     in_size      = in_size,
                                     out_size     = self.encoder_sizes[0],
                                     latent_dim   = latent_lift_proj_dim,
                                     cutoff_den   = cutoff_den,
                                     conv_kernel  = conv_kernel,
                                     filter_size  = filter_size,
                                     lrelu_upsampling  = lrelu_upsampling,
                                     half_width_mult   = half_width_mult,
                                     radial            = radial,
                                     batch_norm        = False,
                                     activation = activation,

                                     is_time = False,
                                     time_steps = time_steps,
                                     lead_time_features = _lead_time_features)

        _out_size = out_size
        if out_size == 1:
            _out_size = in_size

        self.project = LiftProjectBlock(in_channels  = self.encoder_features[0] + self.decoder_features_out[-1],
                                        out_channels = out_dim,
                                        in_size      = self.decoder_sizes[-1],
                                        out_size     = _out_size,
                                        latent_dim   = latent_lift_proj_dim,
                                        cutoff_den   = cutoff_den,
                                        conv_kernel  = conv_kernel,
                                        filter_size  = filter_size,
                                        lrelu_upsampling  = lrelu_upsampling,
                                        half_width_mult   = half_width_mult,
                                        radial            = radial,
                                        batch_norm        = False,
                                        activation = activation,

                                        is_time = False,
                                        time_steps = time_steps,
                                        lead_time_features = _lead_time_features)

        ###################### Define U & D blocks ###################################################################

        self.encoder         = nn.ModuleList([(CNOBlock(in_channels  = self.encoder_features[i],
                                                        out_channels = self.encoder_features[i+1],
                                                        in_size      = self.encoder_sizes[i],
                                                        out_size     = self.encoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,

                                                        is_time = is_time,
                                                        nl_dim = nl_dim,
                                                        time_steps = time_steps,
                                                        lead_time_features = _lead_time_features))
                                              for i in range(self.N_layers)])

        # After the ResNets are executed, the sizes of encoder and decoder might not match (if out_size>1)
        # We must ensure that the sizes are the same, by aplying CNO Blocks
        self.ED_expansion     = nn.ModuleList([(CNOBlock(in_channels = self.encoder_features[i],
                                                         out_channels = self.encoder_features[i],
                                                         in_size      = self.encoder_sizes[i],
                                                         out_size     = self.decoder_sizes[self.N_layers - i],
                                                         cutoff_den   = cutoff_den,
                                                         conv_kernel  = conv_kernel,
                                                         filter_size  = filter_size,
                                                         lrelu_upsampling = lrelu_upsampling,
                                                         half_width_mult  = half_width_mult,
                                                         radial = radial,
                                                         batch_norm = batch_norm,
                                                         activation = activation,

                                                         is_time = is_time,
                                                         nl_dim = nl_dim,
                                                         time_steps = time_steps,
                                                         lead_time_features = _lead_time_features))
                                               for i in range(self.N_layers + 1)])

        self.decoder         = nn.ModuleList([(CNOBlock(in_channels  = self.decoder_features_in[i],
                                                        out_channels = self.decoder_features_out[i],
                                                        in_size      = self.decoder_sizes[i],
                                                        out_size     = self.decoder_sizes[i+1],
                                                        cutoff_den   = cutoff_den,
                                                        conv_kernel  = conv_kernel,
                                                        filter_size  = filter_size,
                                                        lrelu_upsampling = lrelu_upsampling,
                                                        half_width_mult  = half_width_mult,
                                                        radial = radial,
                                                        batch_norm = batch_norm,
                                                        activation = activation,

                                                        is_time = is_time,
                                                        nl_dim = nl_dim,
                                                        time_steps = time_steps,
                                                        lead_time_features = _lead_time_features))
                                              for i in range(self.N_layers)])


        self.decoder_inv    = nn.ModuleList([(CNOBlock(in_channels  =  self.inv_features[i],
                                                       out_channels = self.inv_features[i],
                                                       in_size      = self.decoder_sizes[i],
                                                       out_size     = self.decoder_sizes[i],
                                                       cutoff_den   = cutoff_den,
                                                       conv_kernel  = conv_kernel,
                                                       filter_size  = filter_size,
                                                       lrelu_upsampling = lrelu_upsampling,
                                                       half_width_mult  = half_width_mult,
                                                       radial = radial,
                                                       batch_norm = batch_norm,
                                                       activation = activation,

                                                       is_time = is_time,
                                                       nl_dim = nl_dim,
                                                       time_steps = time_steps,
                                                       lead_time_features = _lead_time_features))
                                             for i in range(self.N_layers + 1)])


        ####################### Define ResNets Blocks ################################################################

        # Here, we define ResNet Blocks.
        # We also define the BatchNorm layers applied BEFORE the ResNet blocks

        # Operator UNet:
        # Outputs of the middle networks are patched (or padded) to corresponding sets of feature maps in the decoder

        self.res_nets = []
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)

        # Define the ResNet blocks & BatchNorm
        for l in range(self.N_layers):
            for i in range(self.N_res):
                self.res_nets.append(ResidualBlock(channels = self.encoder_features[l],
                                                   size     = self.encoder_sizes[l],
                                                   cutoff_den = cutoff_den,
                                                   conv_kernel = conv_kernel,
                                                   filter_size = filter_size,
                                                   lrelu_upsampling = lrelu_upsampling,
                                                   half_width_mult  = half_width_mult,
                                                   radial = radial,
                                                   batch_norm = batch_norm,
                                                   activation = activation,

                                                   is_time = is_time,
                                                   nl_dim = nl_dim,
                                                   time_steps = time_steps,
                                                   lead_time_features = _lead_time_features))
        for i in range(self.N_res_neck):
            self.res_nets.append(ResidualBlock(channels = self.encoder_features[self.N_layers],
                                               size     = self.encoder_sizes[self.N_layers],
                                               cutoff_den = cutoff_den,
                                               conv_kernel = conv_kernel,
                                               filter_size = filter_size,
                                               lrelu_upsampling = lrelu_upsampling,
                                               half_width_mult  = half_width_mult,
                                               radial = radial,
                                               batch_norm = batch_norm,
                                               activation = activation,
                                               is_time = is_time,
                                               nl_dim = nl_dim,
                                               time_steps = time_steps,
                                               lead_time_features = _lead_time_features))

        self.res_nets = torch.nn.Sequential(*self.res_nets)


    def forward(self, x, time, scaling=1.0):

        #Execute Lift ---------------------------------------------------------
        x = self.lift(x, time, scaling)
        skip = []

        # Execute Encoder -----------------------------------------------------
        for i in range(self.N_layers):

            #Apply ResNet & save the result
            #y = x
            for j in range(self.N_res):
                x = self.res_nets[i*self.N_res + j](x, time, scaling)
            skip.append(x)

            # Apply (D) block
            x = self.encoder[i](x, time, scaling)

            #----------------------------------------------------------------------

        # Apply the deepest ResNet (bottle neck)
        for j in range(self.N_res_neck):
            x = self.res_nets[-j-1](x, time, scaling)

        # Execute Decoder -----------------------------------------------------
        for i in range(self.N_layers):

            # Apply (I) block (ED_expansion) & cat if needed
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x, time, scaling) #BottleNeck : no cat
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i], time, scaling)),1)

            if self.add_inv:
                x = self.decoder_inv[i](x, time, scaling)
            # Apply (U) block
            x = self.decoder[i](x, time, scaling)
        # Cat & Execute Projetion ---------------------------------------------

        x = torch.cat((x, self.ED_expansion[0](skip[0], time, scaling)),1)
        x = self.project(x, time, scaling)

        return x


    def get_n_params(self):
        pp = 0

        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def print_size(self):
        nparams = 0
        nbytes = 0

        for param in self.parameters():
            nparams += param.numel()
            nbytes += param.data.element_size() * param.numel()

        print(f'Total number of model parameters: {nparams} (~{format_tensor_size(nbytes)})')

        return nparams
