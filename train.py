from model import Network
from datamodule import FaceDataModule

from pytorch_lightning import Trainer
from argparse import ArgumentParser 

def main(args):
    dm = FaceDataModule(
        data_dir='processed_celeba_small',
        batch_size=32,
        image_size=64,
        num_workers=8
    )

    model = Network(64, 128, 32)

    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)

    main(parser.parse_args())