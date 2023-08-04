

import argparse

from diffusion_base import *

from utils import *


def main(config):

    unet_params = config["unet_params"]
    model = Unet(**unet_params).cuda()

    diff_params = config["diff_params"]
    diffusion = GaussianDiffusion(model, **diff_params).cuda()

    trainer_params = config["trainer_params"]
    trainer = Trainer(diffusion, **trainer_params, wandb_config=config)

    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
    )
    args = parser.parse_args()
    config = load_config_file(args.config)

    main(config)
