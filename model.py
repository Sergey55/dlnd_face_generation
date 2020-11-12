import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transforms import Scale

class Net(pl.LightningModule):
    """ Network for generating people faces"""

    def __init__(self, g_conv_dim, z_size, d_conv_dim):
        super(Net, self).__init__()

        self.z_size = z_size

        self.generator = Generator(z_size, g_conv_dim)
        self.discriminator = Discriminator(d_conv_dim)

        self.criterion = nn.BCEWithLogitsLoss()

        self.scale = Scale()

    def forward(self, count):
        z = torch.rand(size=(count, self.z_size), device=self._device).uniform_(0, 1)
        fake_images = self.generator(z)

        return fake_images

    def real_loss(self, D_out, smooth = False):
        """ Calculate how close discriminator outputs are to being real.

        Args:
            D_out:          discriminator logits.
            smooth:         smooth labels or not

        Returns:
            real loss.
        """
        batch_size = D_out.size(0)
        labels = torch.ones(batch_size, device=self._device)

        if smooth:
            labels = labels * 0.9

        loss = self.criterion(D_out.squeeze(), labels)

        return loss

    def fake_loss(self, D_out):
        """
            Calculate how close discriminator oututs are to being fake.

        Args:
            D_out:          discriminator logits.

        Returns:
            fake loss
        """
        batch_size = D_out.size(0)

        labels = torch.zeros(batch_size, device=self._device)

        loss = self.criterion(D_out.squeeze(), labels)

        return loss

    def configure_optimizers(self):
        lr = 0.0005
        beta1 = 0.3
        beta2 = 0.999

        d_optimizer = optim.Adam(self.discriminator.parameters(), lr, [beta1, beta2]) 
        g_optimizer = optim.Adam(self.generator.parameters(), lr, [beta1, beta2])

        return [d_optimizer, g_optimizer]

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Train the discriminator
        if optimizer_idx == 0:
            return self.train_discriminator(batch[0])

        # Train the generator:
        if optimizer_idx == 1:
            return self.train_generator(batch[0])

    def train_discriminator(self, batch):
            batch_size = batch.size(0)

            real_images = self.scale(batch)
        
            d_real_out = self.discriminator(real_images)
            d_real_loss = self.real_loss(d_real_out)

            z = torch.rand(size=(batch_size, self.z_size), device=self._device).uniform_(0, 1)
            fake_images = self.generator(z)

            d_fake_out = self.discriminator(fake_images)
            d_fake_loss = self.fake_loss(d_fake_out)

            loss = d_real_loss + d_fake_loss

            self.log('discriminator_loss', loss, on_step=True, on_epoch=True)

            return {'loss': loss}

    def train_generator(self, batch):
            batch_size = batch.size(0)

            z = torch.rand(size=(batch_size, self.z_size), device=self._device).uniform_(0, 1)
            fake_images = self.generator(z)

            d_fake_out = self.discriminator(fake_images)

            loss = self.real_loss(d_fake_out, True)

            self.log('generator_loss', loss, on_step=True, on_epoch=True)

            return {'loss': loss}

class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        
        self.conv1 = self.conv(3, conv_dim, 4, batch_norm=False)
        #out 16 x 16 x conv_dim
        self.conv2 = self.conv(conv_dim, conv_dim * 2, 4)
        #out 8 x 8 x conv_dim * 2
        self.conv3 = self.conv(conv_dim * 2, conv_dim * 4, 4)
        #out 4 x 4 x conv_dim * 4
        self.conv4 = self.conv(conv_dim * 4, conv_dim * 8, 4)
        #out 2 x 2 x conv_dim * 8
        self.conv5 = self.conv(conv_dim * 8, conv_dim * 16, 4)
        #out 1 x 1 x conv_dim * 16
        
        self.fc = nn.Linear(conv_dim * 16, 1)
        

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.leaky_relu(self.conv5(x), 0.2)
        
        x = x.view(-1, self.conv_dim * 16)
        
        x = self.fc(x)
        
        return x

    def conv(self, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        """Create Sequence containing convolutional layer with optional batch normalization layer

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kerne_size: Kernel size
            stride: Stride (Default: 2)
            padding: Padding (Defailt: 1)
            batch_norm: Add optional batch normalization layer
        Return:
            nn.Sequence containing Conv layer and (optional) BatchNorm layer
        """
        layers = []

        layers.append(nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                bias=False))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)

class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()
        
        self.z_size = z_size
        self.conv_dim = conv_dim

        self.fc = nn.Linear(self.z_size, self.conv_dim * 8 * 2 * 2)
        
        # complete init function
        self.deconv1 = self.deconv(self.conv_dim * 8, self.conv_dim * 4, 4)
        self.deconv2 = self.deconv(self.conv_dim * 4, self.conv_dim * 2, 4)
        self.deconv3 = self.deconv(self.conv_dim * 2, self.conv_dim, 4)
        self.deconv4 = self.deconv(self.conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = x.view(-1, self.conv_dim * 8, 2, 2)
        
        x = F.leaky_relu(self.deconv1(x))
        x = F.leaky_relu(self.deconv2(x))
        x = F.leaky_relu(self.deconv3(x))
        
        x = F.tanh(self.deconv4(x))
        
        return x

    def deconv(self, in_channels, out_channels, kernel_size,  stride=2, padding=1, batch_norm=True):
        """Create a nn.Sequence containing Tranposed-Convoltional layer and optional BatchNormalization layer.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kerne_size: Kernel size
            stride: Stride (Default: 2)
            padding: Padding (Defailt: 1)
            batch_norm: Add optional batch normalization layer

        Return:
            nn.Sequence containing Tranposed-Convoltional layer and (optional) BatchNorm layer
        """
        layers = []

        layers.append(nn.ConvTranspose2d(in_channels,
                                        out_channels,
                                        kernel_size,
                                        stride,
                                        padding,
                                        bias=False))

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        return nn.Sequential(*layers)