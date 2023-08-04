
import os
import argparse

import wandb
from tqdm import tqdm

import torch

from utils import *
from dataset import *


def train(args,
          epoch,
          model,
          train_dataloader,
          optimizer,
          loss_fn,
          grad_scaler,
          wandb_logger,
          ) -> None:
    """ Main training function for the DifFIQA(R) approach.

    Args:
        args (Argument): Arguments from the training script.
        model (torch.nn.Module): Model to be trained.
        train_dataloader (torch.utils.data.DataLoader): Dataloader of training samples.
        optimizer (torch.optim.Optimizer): Optimizer used.
        loss_fn (torch.nn.loss._Loss): Loss used.
        grad_scaler (torch.cuda.amp.GradScaler): Amp gradient scaler.
        wandb_logger (wandb.logger): Wandb logger.
    """

    model.train()
    for (image_batch, label_batch) in (pbar := tqdm(train_dataloader, 
                                           desc=f" Training Epoch ({epoch}/{args.base.epochs}), Loss: NaN ", 
                                           disable=not args.base.verbose)):

        image_batch, label_batch = image_batch.to(args.base.device), label_batch.to(args.base.device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out_batch = model(image_batch).squeeze()
            loss = loss_fn(out_batch, label_batch)

        grad_scaler.scale(loss).backward() 
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        grad_scaler.step(optimizer) 
        grad_scaler.update()

        pbar.set_description(f" Training Epoch ({epoch}/{args.base.epochs}), Loss: {loss.item():.4f} ")

        if args.wandb.use:
            wandb_logger.log({"train_loss": loss.item()})


@torch.no_grad()
def validate(args,
             best_val_loss,
             model,
             val_dataloader,
             loss_fn,
             wandb_logger) -> float:
    """ Main validation function for the DifFIQA(R) approach.

    Args:
        args (Argument): Arguments from the training script.
        best_val_loss (float): Best recorded validation loss.
        model (torch.nn.Module): Model to be trained.
        val_dataloader (torch.utils.data.DataLoader): Dataloader of validation samples.
        loss_fn (torch.nn.loss._Loss): Loss used.
        wandb_logger (wandb.logger): Wandb logger.

    Returns:
        float: Best recored validation loss.
    """

    per_epoch_val_loss = 0.
    model.eval()
    for (image_batch, label_batch) in tqdm(val_dataloader, 
                                           desc=" Validation ", 
                                           disable=not args.base.verbose):

        image_batch, label_batch = image_batch.to(args.base.device), label_batch.to(args.base.device)

        out_batch = model(image_batch).detach().squeeze()

        loss = loss_fn(out_batch, label_batch)
        per_epoch_val_loss += loss.item()

    per_epoch_val_loss = per_epoch_val_loss / len(val_dataloader)

    if args.wandb.use:
        wandb_logger.log({"val_loss": per_epoch_val_loss})

    if per_epoch_val_loss < best_val_loss:
        torch.save(model.state_dict(), os.path.join(args.base.save_path, f"model.pth"))
        best_val_loss = per_epoch_val_loss
    
    return best_val_loss


def main(args):

    # Seed all libraries to ensure consistency between runs.
    seed_all(args.base.seed)

    # Check if save path valid
    assert os.path.exists(args.base.save_path), f"Path {args.base.save_path} does not exist"
    #f_save = os.path.join(args.base.save_path, f"{args.face_embedder}-{args.batch_size}")

    # Load the training FR model and construct the transformation
    model, trans = construct_full_model(args.model.config)
    model.to(args.base.device)

    # Construct validation and training dataloaders
    train_dataset, val_dataset = construct_datasets(args.dataset, trans)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   **args_to_dict(args.dataloader.train.params, {}))
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 **args_to_dict(args.dataloader.val.params, {}))

    # Create optimizer from config
    optimizer = construct_optimizer(args.optimizer, model)

    # Load desired loss function
    loss_fn = load_module(args.loss)

    # Code uses torch.amp
    grad_scaler = torch.cuda.amp.GradScaler()

    # Construct WANDB logger if so requested
    wandb_logger = None
    if args.wandb.use:
        wandb_logger = wandb.init(project=args.wandb.project, config={"args": args_to_dict(args, {})})

    # Train loop
    best_val_loss = float("inf")
    for epoch in range(args.base.epochs):

        train(args, epoch, model, train_dataloader, optimizer, loss_fn, grad_scaler, wandb_logger)

        best_val_loss = validate(args, best_val_loss, model, val_dataloader, loss_fn, wandb_logger)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", 
        "--config",
        type=str,
        help=' Location of the DifFIQA(R) training configuration. '
    )
    args = parser.parse_args()
    arguments = parse_config_file(args.config)

    main(arguments)