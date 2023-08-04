
import math
from pathlib import Path
from random import sample
from multiprocessing import cpu_count

import torch
from torch.utils.data import DataLoader

from torch.optim import Adam
from torchvision.utils import save_image

from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from .diff_util import *
from .diff_dataset import *
from .diff_unet import *
from .diff_gauss import *

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        preload_unet = None,
        use_wandb = False,
        project_name = "",
        wandb_config = {},
        verbose: False
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            log_with='wandb' if use_wandb else None
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = DiffDataset(folder, self.image_size)
        dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        # choose some sampling images 

        self.sampling_imgs = torch.tensor([])
        for idx in sample(range(0, len(self.ds)), k=25):
            lq_img, _ = self.ds[idx] 
            self.sampling_imgs = torch.cat((self.sampling_imgs, lq_img.unsqueeze(0)), dim=0)
        self.sampling_imgs = normalize_to_neg_one_to_one(self.sampling_imgs)
        self.sampling_imgs = [self.sampling_imgs[self.batch_size * i: min(self.batch_size * (i + 1), self.num_samples)] for i in range((self.sampling_imgs.shape[0] // self.batch_size) + 1)]

        self.sampling_times = torch.tensor([self.model.T]).long()

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        if preload_unet:
            self.preload_unet(preload_unet)

        self.use_wandb = use_wandb
        self.verbose = verbose

        if self.use_wandb:
            self.accelerator.init_trackers(project_name=project_name, 
                                           config=wandb_config)


    def preload_unet(self, location):
        
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(location, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])


    def save(self, milestone, loss):
        if not self.accelerator.is_local_main_process:
            return
        
        self.best_loss = loss

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))


    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        if isinstance(milestone, int):
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        else:
            data = torch.load(milestone, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self):

        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, 
                  total = self.train_num_steps, 
                  disable = (not accelerator.is_main_process) and self.verbose) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    
                    lq_imgs, hq_imgs = next(self.dl)
                    lq_imgs, hq_imgs = lq_imgs.to(device), hq_imgs.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(lq_imgs, hq_imgs)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                if self.use_wandb:
                    accelerator.log({"total_loss": total_loss})

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_images_list = list(map(lambda n, imgs: self.ema.ema_model.sample(batch_size=n, imgs=imgs, times=self.sampling_times, nof=self.sampling_times.item() + 1), batches, self.sampling_imgs))
                                                                                                           
                        all_images = torch.cat(all_images_list, dim = 0)
                        save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), 
                                   nrow = int(math.sqrt(self.num_samples)), normalize=True, value_range=(-1., 1.))
                        self.save(milestone, total_loss)

                pbar.update(1)

        accelerator.print('training complete')
