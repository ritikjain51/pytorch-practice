import os
import torch
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torchvision import transforms

import config
from data import get_dataloader, read_files
from model import ImageClassificationModel
from trainer import Trainer

# Image Transformation using torch vision
transform = transforms.Compose(
    [
    transforms.Resize(config.IMG_DIM),
    transforms.RandomAutocontrast(0.2),
    transforms.RandomRotation(270),
    transforms.RandomVerticalFlip(0.4),
    transforms.RandomHorizontalFlip(0.2),
    transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize(config.IMG_DIM),
        transforms.Normalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
)

# Get Dataloader
# Loading the train dataset
train_path = os.path.join(config.BASE_DATASET_PATH, "seg_train")
train_x, train_y = read_files(train_path, format="jpg")
# Training OneHotEncoder
if os.path.exists(config.TARGET_TRANSFORMER_PATH):
    target_enc = joblib.load(config.TARGET_TRANSFORMER_PATH)
else:
    target_enc = LabelEncoder()
    target_enc.fit(train_y)
    joblib.dump(target_enc, config.TARGET_TRANSFORMER_PATH)

train_loader, train_dataset = get_dataloader(x=train_x, y=train_y, batch_size=config.BATCH_SIZE, transform=transform, target_transform=target_enc, channel_first=True, device=config.DEVICE)


# Loading Test Dataset ...
test_path = os.path.join(config.BASE_DATASET_PATH, "seg_test")
test_x, test_y = read_files(test_path, format="jpg")
test_loader, test_dataset = get_dataloader(x=test_x, y=test_y, batch_size=config.BATCH_SIZE, transform=test_transform, target_transform=target_enc, channel_first=True, device=config.DEVICE)



# Initializing Model 
# model = ImageClassificationModel(num_classes = config.NUM_CLASSES)
model = torch.load(f"{config.MODEL_PATH}_9.pt")
model.to(config.DEVICE)


# Optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#scheduler
sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1)

# Loss function (criteria)
criteria = torch.nn.CrossEntropyLoss()

# Start Model training 
trainer_class = Trainer(optimizer=optimizer, criteria=criteria, epochs=10, scheduler=sch)


if __name__ == "__main__":
    trainer_class.fit(model, train_loader=train_loader, valid_loader=test_loader, device=config.DEVICE, model_path = config.MODEL_PATH)