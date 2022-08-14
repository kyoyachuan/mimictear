import os


ROOT = '/home/kyoyachuan/mimictear'
CHECKPOINT_ROOT = os.path.join(ROOT, 'checkpoints')
IMAGES_ROOT = os.path.join(ROOT, 'images')
METADATAS_ROOT = os.path.join(ROOT, 'metadatas')
EVALUATOR_PATH = os.path.join(METADATAS_ROOT, 'checkpoint.pth')
LABEL_PATH = os.path.join(METADATAS_ROOT, 'objects.json')
TRAIN_PATH = os.path.join(METADATAS_ROOT, 'train.json')

EVAL_ITERS = 30


class LossType:
    MINIMAX = 'minimax'
    WASSERSTEIN = 'wasserstein'
    HINGE = 'hinge'
