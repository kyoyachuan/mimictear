import os

import numpy as np
import torch
from torch import optim
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
from wandb import AlertLevel

from .contants import LossType, CHECKPOINT_ROOT, EVAL_ITERS
from .losses import get_loss
from .dataloader import random_generate_labels


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        data_loader,
        evaluator,
        test_labels,
        trainer_cfg,
        generator_cfg,
        discriminator_cfg,
        wandb_cfg
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.data_loader = data_loader
        self.evaluator = evaluator
        self.test_labels = test_labels
        self.trainer_cfg = trainer_cfg
        self.generator_cfg = generator_cfg
        self.discriminator_cfg = discriminator_cfg
        self.wandb_cfg = wandb_cfg

        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=trainer_cfg.lr_g, betas=(0., 0.9))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=trainer_cfg.lr_d, betas=(0., 0.9))
        self.loss = get_loss(trainer_cfg.loss_type)

        self.checkpoint_path = os.path.join(CHECKPOINT_ROOT, trainer_cfg.checkpoint_path)
        self.init_wandb()
        self.init_checkpoint_dir()
        self.best_accuracy = 0.0
        self.best_epoch = 0

    def init_wandb(self):
        wandb.init(
            project=self.wandb_cfg.project,
            entity=self.wandb_cfg.entity,
            name=self.wandb_cfg.name,
            config={**self.trainer_cfg, **self.generator_cfg, **self.discriminator_cfg},
        )

    def init_checkpoint_dir(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def save_model(self, file_str, epoch=None):
        torch.save({
            "generator": self.generator,
            "discriminator": self.discriminator,
            "g_optimizer": self.g_optimizer,
            "d_optimizer": self.d_optimizer,
            "epoch": epoch,
            "best_accuracy": self.best_accuracy,
            "best_epoch": self.best_epoch
        }, f"{self.checkpoint_path}/{file_str}.pth")

    def train(self):
        for epoch in range(self.trainer_cfg.niters):
            mean_d_loss, mean_g_loss = self.train_step(epoch)

            self.generator.train(False)
            accuracy = 0
            for _ in range(EVAL_ITERS):
                imgs, labels = self.generate_test_images()
                acc = self.evaluator.eval(imgs, labels)
                accuracy += acc
            accuracy /= EVAL_ITERS
            self.generator.train(True)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_epoch = epoch

                wandb.alert(
                    title=f"New best generated accuracy",
                    text=f"Accuracy: {accuracy:.4f} at epoch {epoch}",
                    level=AlertLevel.INFO,
                )
                self.save_model('best', epoch)

            wandb.log({"d_loss": mean_d_loss, "g_loss": mean_g_loss, "accuracy": accuracy}, step=epoch)

            if epoch % self.trainer_cfg.log_image_interval == 0 or epoch == self.trainer_cfg.niters - 1:
                wandb.log({"generated_images": wandb.Image(np.transpose(
                    make_grid(imgs, value_range=(-1, 1)).cpu().numpy(), (1, 2, 0)))}, step=epoch)

            if epoch % self.trainer_cfg.save_interval == 0 or epoch == self.trainer_cfg.niters - 1:
                self.save_model(epoch, epoch)

    def train_step(self, epoch):
        pbar = tqdm(range(self.trainer_cfg.epoch_size))
        pbar.set_description(f"Epoch {epoch}/{self.trainer_cfg.niters}")

        total_d_loss = 0
        total_g_loss = 0
        g_loss_item = 0
        requires_grad(self.generator, False)
        requires_grad(self.discriminator, True)

        for i in pbar:
            self.discriminator.zero_grad()
            real_image, label = next(self.data_loader)
            b_size = real_image.size(0)
            real_image = real_image.cuda()
            label = label.cuda()
            fake_image = self.generator(
                torch.randn(b_size, self.generator_cfg.code_dim).cuda(), label
            )
            fake_predict = self.discriminator(fake_image, label)
            real_predict = self.discriminator(real_image, label)
            d_loss = self.loss.d_loss(real_predict, fake_predict)
            d_loss_item = d_loss.detach().cpu().item()
            d_loss.backward()
            self.d_optimizer.step()
            if self.trainer_cfg.loss_type == LossType.WASSERSTEIN:
                self.loss.clamp_params(self.discriminator.parameters(), self.trainer_cfg.clamp_values)

            total_d_loss += d_loss_item

            if (i + 1) % self.trainer_cfg.n_d == 0:
                self.generator.zero_grad()
                requires_grad(self.generator, True)
                requires_grad(self.discriminator, False)
                input_class = random_generate_labels(self.trainer_cfg.batch_size, self.test_labels.size(1)).cuda()
                fake_image = self.generator(
                    torch.randn(self.trainer_cfg.batch_size, self.generator_cfg.code_dim).cuda(), input_class
                )
                predict = self.discriminator(fake_image, input_class)
                g_loss = self.loss.g_loss(predict)
                g_loss_item = g_loss.detach().cpu().item()
                g_loss.backward()
                self.g_optimizer.step()
                requires_grad(self.generator, False)
                requires_grad(self.discriminator, True)

                total_g_loss += g_loss_item

            pbar.set_postfix(d_loss=d_loss_item, g_loss=g_loss_item)

        return total_d_loss / self.trainer_cfg.epoch_size, total_g_loss / (
            self.trainer_cfg.epoch_size / self.trainer_cfg.n_d)

    def generate_test_images(self):
        input_class = self.test_labels.cuda()
        fake_images = self.generator(
            torch.randn(len(self.test_labels), self.generator_cfg.code_dim).cuda(), input_class
        )
        return fake_images.detach(), input_class.detach()
