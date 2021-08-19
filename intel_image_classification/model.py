import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class ImageClassificationModel(nn.Module):

    def __init__(self, num_classes):
        super(ImageClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.base_model = resnet34(pretrained=True, progress=True)
        self.disable_base_model_training()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1000, num_classes)
    
    def disable_base_model_training(self):
        for x in self.base_model.parameters():
            x.requires_grad_(False)

    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return F.softmax(self.fc(x))

# if __name__ == "__main__":
    # model = ImageClassificationModel(6)
    # print(model)
