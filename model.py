import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools
from utils import *

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=32, max_dim=512, attr_dim=18, z_dim=50):
        super(Generator, self).__init__()

        self.encode = self.encoder_net(conv_dim, max_dim)
        self.decode = self.decoder_net(max_dim, attr_dim + z_dim, conv_dim)

    def encoder_net(self, conv_dim, max_dim):
         
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim*2, affine=True))
        layers.append(nn.ReLU(inplace=True))

        curr_in = conv_dim * 2

        for i in range(4):
            out_mult_1 = 2 + i*2
            out_mult_2 = 2 + i
            out_mult_3 = 4 + i*2

            layers.append(nn.Conv2d(curr_in, conv_dim*out_mult_1, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*out_mult_1, affine=True))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(conv_dim*out_mult_1, conv_dim*out_mult_2, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*out_mult_2, affine=True))
            layers.append(nn.ReLU(inplace=True))

            layers.append(nn.Conv2d(conv_dim*out_mult_2, conv_dim*out_mult_3, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*out_mult_3, affine=True))
            layers.append(nn.ReLU(inplace=True))

            curr_in = conv_dim*out_mult_3

        layers.append(nn.AdaptiveAvgPool2d((1,1)))

        return nn.Sequential(*layers)

    def decoder_net(self,max_dim, c_dim, conv_dim):
        curr_dim = conv_dim

        layers = []
        layers.append(nn.ConvTranspose2d(conv_dim*10 + c_dim, conv_dim*10, kernel_size=6, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim*10, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose2d(conv_dim*10, conv_dim*5, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim*5, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose2d(conv_dim*5, conv_dim*8, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim*8, affine=True))
        layers.append(nn.ReLU(inplace=True))
        
        curr_dim = conv_dim*8
        for i in reversed(range(3)):
            mult_3 = 2 + i*2
            mult_2 = 2 + i
            mult_1 = 4 + i*2

            layers.append(nn.ConvTranspose2d(curr_dim, conv_dim*mult_1, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*mult_1, affine=True))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.ConvTranspose2d(conv_dim*mult_1, conv_dim*mult_2, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*mult_2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.ConvTranspose2d(conv_dim*mult_2, conv_dim*mult_3, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*mult_3, affine=True))
            layers.append(nn.ReLU(inplace=True))

            curr_dim = conv_dim*mult_3

        layers.append(nn.ConvTranspose2d(curr_dim, conv_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim*2, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose2d(conv_dim*2, conv_dim, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.ConvTranspose2d(conv_dim, 3, kernel_size=3, stride=1, padding=1, bias=True))
        layers.append(nn.Tanh())

        return nn.Sequential(*layers)


    def forward(self, x, attr_vec, z_vec):
        enc = self.encode(x)
        attr_vec = attr_vec.unsqueeze(-1).unsqueeze(-1)
        z_vec = z_vec.unsqueeze(-1).unsqueeze(-1)
        concat = torch.cat([attr_vec, enc, z_vec], dim=1)
        dec = self.decode(concat)
        return enc, concat, dec


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=96, conv_dim=64, attr_dim=18, iden_dim=1000, repeat_num=4):
        super(Discriminator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim, affine=True))
        layers.append(nn.LeakyReLU(0.2,inplace=True))

        layers.append(nn.Conv2d(conv_dim, conv_dim*2, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(conv_dim*2, affine=True))
        layers.append(nn.LeakyReLU(0.2,inplace=True))

        curr_in = conv_dim * 2

        for i in range(4):
            out_mult_1 = 2 + i*2
            out_mult_2 = 2 + i
            out_mult_3 = 4 + i*2

            layers.append(nn.Conv2d(curr_in, conv_dim*out_mult_1, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*out_mult_1, affine=True))
            layers.append(nn.LeakyReLU(0.2,inplace=True))

            layers.append(nn.Conv2d(conv_dim*out_mult_1, conv_dim*out_mult_2, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*out_mult_2, affine=True))
            layers.append(nn.LeakyReLU(0.2,inplace=True))

            layers.append(nn.Conv2d(conv_dim*out_mult_2, conv_dim*out_mult_3, kernel_size=4, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(conv_dim*out_mult_3, affine=True))
            layers.append(nn.LeakyReLU(0.2,inplace=True))

            curr_in = conv_dim*out_mult_3

        layers.append(nn.AdaptiveAvgPool2d((1,1)))

        self.main = nn.Sequential(*layers)

        self.real_classifier = nn.Conv2d(curr_in, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.attr_classifier = nn.Conv2d(curr_in, attr_dim, kernel_size=1, padding=0, bias=True)
        self.iden_classifier = nn.Conv2d(curr_in, iden_dim, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        h = self.main(x)

        out_real = self.real_classifier(h)
        out_class = self.attr_classifier(h)
        out_iden = self.iden_classifier(h)
        return out_real.squeeze(), out_class.squeeze(), out_iden.squeeze()