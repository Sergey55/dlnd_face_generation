from model import Net
from datamodule import FaceDataModule

from pytorch_lightning import Trainer
from argparse import ArgumentParser

from weights_initializer import WeightsInitializer

def main(args):
    dm = FaceDataModule(
        data_dir='processed_celeba_small',
        batch_size=128,
        image_size=64,
        num_workers=8
    )

    model = Net(g_conv_dim=64, z_size=128, d_conv_dim=32)

    weights_initializer = WeightsInitializer()

    # Init weights
    model.discriminator.apply(weights_initializer.init_weights_kaiming)
    model.generator.apply(weights_initializer.init_weights_kaiming)
    model.generator.deconv4.apply(weights_initializer.init_weights_xavier)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)

    main(parser.parse_args())