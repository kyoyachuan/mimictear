import os
import random

import hydra
from omegaconf import DictConfig
import torch
from torchvision.utils import save_image

from mimictear.dataloader import get_test_label
from mimictear.evaluator import Evaluation
from mimictear.contants import CHECKPOINT_ROOT


torch.backends.cudnn.benchmark = True


@hydra.main(config_path='configs', config_name='')
def main(cfg: DictConfig) -> None:
    random.seed(cfg.base.seed)
    torch.manual_seed(cfg.base.seed)
    torch.cuda.manual_seed_all(cfg.base.seed)

    test_labels = get_test_label(cfg.trainer.test_labels_path).cuda()

    checkpoint_path = os.path.join(CHECKPOINT_ROOT, cfg.trainer.checkpoint_path)
    model_path = os.path.join(checkpoint_path, 'best.pth')
    image_path = os.path.join(checkpoint_path, 'demo.png')

    generator = torch.load(model_path)['generator']
    evaluator = Evaluation()

    fake_images = generator(
        torch.randn(len(test_labels), cfg.generator.code_dim).cuda(), test_labels
    )

    accuracy = evaluator.eval(fake_images, test_labels)
    save_image(fake_images + 0.5, image_path, padding=2)
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main()
