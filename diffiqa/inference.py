
import os
import pickle
import argparse

import torch
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

from utils import *
from dataset import *
from diffusion_base import *


def load_diff(config: dict, disable_wandb: bool=True) -> Trainer:
    """ Generate the diffusion module used during training.

    Args:
        config (dict): Config of training parameters.
        disable_wandb (bool, optional): Disable wandb for inference. Defaults to True.

    Returns:
        Trainer: Diffusion trainer class of trained model.
    """

    train_config = load_config_file(config["train_config"])

    unet_params = train_config["unet_params"]
    model = Unet(**unet_params).cuda()

    diff_params = train_config["diff_params"]
    diffusion = GaussianDiffusion(model, **diff_params).cuda()

    trainer_params = train_config["trainer_params"]
    trainer_params["use_wandb"] = 0 if disable_wandb else 1
    trainer = Trainer(diffusion, **trainer_params)

    return trainer


@torch.no_grad()
def calc_score(a: torch.tensor, b: torch.tensor, sim_f: torch.nn.Module, repeats: int) -> np.array:
    """ Calculates similarities between embeddings in a and b using similarity function sim_f.

    Args:
        a (torch.tensor): First tensor of embeddings.
        b (torch.tensor): Second tensor of embeddings.
        sim_f (torch.nn.Module): Similarity function.
        repeats (int): Number of repeats used.

    Returns:
        np.array: Numpy array of mean similarity values.
    """

    sims = sim_f(a, b)
    sims_unpacked = rearrange(sims, "(b n) -> b n", n=repeats)
    sims_mean = torch.mean(sims_unpacked, dim=1).cpu().numpy()

    return sims_mean


@torch.no_grad()
def inference(config):

    assert os.path.exists(config["save_loc"]), f"Save path {config['save_loc']} does not exist!"

    # Load diffusion module
    trainer = load_diff(config)
    trainer.load(os.path.join(config["model_loc"]))

    device = trainer.accelerator.device

    cossim = torch.nn.CosineSimilarity()

    # Load the face recognition model
    fr_config = load_config_file(config["fr_config"])
    fr_model, trans = construct_full_model(fr_config)
    fr_model.eval().to(device)

    # Define base parameters used
    used_timestep = config["used_timesteps"]
    repeats = config["repeats"]
    _batch_size = config["batch_size"] // (2 * repeats)

    # Construct dataloader
    dataset = ImageDataset(config["images_loc"], trans)
    dataloader = DataLoader(dataset, batch_size=_batch_size, pin_memory=True)

    quality_scores = {}

    t = torch.tensor([used_timestep]).long().cuda()

    for (name_batch, img_batch, flip_img_batch) in tqdm(dataloader,
                                                        desc=f" Inference ",
                                                        disable=not config["verbose"]):
        
        # Combine base and flipped images and repeat them 
        starting_imgs = torch.cat((img_batch, flip_img_batch), dim=0).detach().clone().to(device)
        starting_imgs = repeat(starting_imgs, 'b c h w -> (b n) c h w', n=repeats)

        # Apply forward and backward diffusion to images
        noisy_imgs = trainer.model.q_sample(starting_imgs, t, None)
        reconstructed_imgs = trainer.model.ddim_sample_at_t((starting_imgs.shape[0], 3, 112, 112), 
                                                            noisy_imgs, t=t, nof_s=used_timestep+1, 
                                                            verbose=False)

        # Run all images through the fr model
        with torch.no_grad():
            outs = fr_model(torch.cat((starting_imgs, noisy_imgs, reconstructed_imgs), dim=0))

        # Separate the embeddings
        starting_base_outs, starting_flip_outs, \
        noisy_base_outs, noisy_flip_outs, \
        recon_base_outs, recon_flip_outs = outs.chunk(6)
        # starting_base_outs = outs[:outs.shape[0] // 6]
        # starting_flip_outs = outs[outs.shape[0] // 6: 2 * (outs.shape[0] // 6)]
        # noisy_base_outs = outs[2 * (outs.shape[0] // 6): 3 * (outs.shape[0] // 6)]
        # noisy_flip_outs = outs[3 * (outs.shape[0] // 6): 4 * (outs.shape[0] // 6)]
        # recon_base_outs = outs[4 * (outs.shape[0] // 6): 5 * (outs.shape[0] // 6)]
        # recon_flip_outs = outs[5 * (outs.shape[0] // 6):]

        # Calculate the quality scores
        qs_scores = 1./5. * \
                    (calc_score(starting_base_outs, noisy_base_outs, cossim, repeats) + \
                    calc_score(starting_base_outs, recon_base_outs, cossim, repeats) + \
                    calc_score(starting_base_outs, noisy_flip_outs, cossim, repeats) + \
                    calc_score(starting_base_outs, recon_flip_outs, cossim, repeats) + \
                    calc_score(starting_base_outs, starting_flip_outs, cossim, repeats))

        quality_scores.update(zip(name_batch, qs_scores))
        
    # Save results
    with open(os.path.join(config["save_loc"], "quality_scores.pkl"), "wb") as pkl_out:
        pickle.dump(quality_scores, pkl_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
    )
    args = parser.parse_args()

    config = load_config_file(args.config)
    inference_config = config["inference"]

    inference(inference_config)

