import random

import hydra
from omegaconf import DictConfig
import torch

from mimictear.dataloader import get_test_label, get_data
from mimictear.model import Generator, Discriminator
from mimictear.trainer import Trainer
from mimictear.evaluator import Evaluation


torch.backends.cudnn.benchmark = True


@hydra.main(config_path='configs', config_name='')
def main(cfg: DictConfig) -> None:
    random.seed(cfg.base.seed)
    torch.manual_seed(cfg.base.seed)
    torch.cuda.manual_seed_all(cfg.base.seed)

    dataloader = iter(get_data(cfg.trainer.batch_size, cfg.base.num_workers))
    test_labels = get_test_label(cfg.trainer.test_labels_path)

    generator = Generator(
        code_dim=cfg.generator.code_dim,
        cond_dim=cfg.generator.cond_dim,
        n_class=test_labels.size(1),
        self_attention=cfg.generator.self_attention,
        cbn=cfg.generator.cbn
    )
    discriminator = Discriminator(
        n_class=test_labels.size(1),
        self_attention=cfg.discriminator.self_attention,
        projection=cfg.discriminator.projection,
    )

    evaluator = Evaluation()

    trainer = Trainer(
        generator=generator.cuda(),
        discriminator=discriminator.cuda(),
        data_loader=dataloader,
        evaluator=evaluator,
        test_labels=test_labels,
        trainer_cfg=cfg.trainer,
        generator_cfg=cfg.generator,
        discriminator_cfg=cfg.discriminator,
        wandb_cfg=cfg.wandb,
    )

    trainer.train()


if __name__ == '__main__':
    main()
