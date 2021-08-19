import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")
BATCH_SIZE = 256
IMAGE_SIZE=150
BASE_DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset") 
CLASS_PATH = os.path.join(os.path.dirname(__file__), "classes.json")
TARGET_TRANSFORMER_PATH = os.path.join(os.path.dirname(__file__), "target_transform.pkl")
NUM_CLASSES = 6
IMG_DIM=(150, 150)