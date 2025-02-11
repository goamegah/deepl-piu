import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(Path(ROOT_DIR).parent.absolute(), 'dataset')
CHECKPOINT_PATH = os.path.join(Path(ROOT_DIR).parent.absolute(), 'checkpoints')
TRAIN_DATA_PATH = os.path.join(DATASET_PATH, 'train.csv')
TEST_DATA_PATH = os.path.join(DATASET_PATH, 'test.csv')

NUM_CLASSES = 4

print(DATASET_PATH)