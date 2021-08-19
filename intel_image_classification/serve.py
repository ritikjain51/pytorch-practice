"""
torch-model-archiver --model-name waveglow_synthesizer --version 1.0 --model-file waveglow_model.py 
--serialized-file nvidia_waveglowpyt_fp32_20190306.pth --handler waveglow_handler.py --extra-files tacotron.zip,nvidia_tacotron2pyt_fp32_20190306.pth
"""
"""
torch-model-archiver --model-name resnet34_image_classifier --version 1.0 --model-file model.py --serialized-file model_9.pt --handler serve.py --requirements-file requirements.txt --extra-files target_transform.pkl
>torch-model-archiver --model-name resnet34_image_classifier --version 1.0 --model-file model.py --serialized-file jit_model.pt --handler serve.py --requirements-file requirements.txt --extra-files index_to_name.json -f
>torchserve --model-store . --models resnet34_image_classifier.mar --start
"""

import logging
import os
import io
from PIL import Image
import torch
from base64 import b64decode
import numpy as np
from torchvision.transforms import transforms
from ts.torch_handler.vision_handler import VisionHandler
from ts.utils.util import map_class_to_label, load_label_mapping

logger = logging.getLogger(__name__)

class ImageHandler(VisionHandler):

    def __init__(self):
        super(ImageHandler, self).__init__()
        self.model = None
        self.target_transform = None
        self.content = None

    def image_processing(self, img):
        test_transform = transforms.Compose(
            [
                transforms.Resize((150,150)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return test_transform(img)

    def preprocess(self, data):
        try:
            images = []
            for row in data:
                image = row.get("body", row.get("data"))
                if isinstance(image, str) and not image.__contains__("http"):
                    image = b64decode(image)
                
                if isinstance(image, (bytearray, bytes)):
                    image = Image.open(io.BytesIO(image))
                
                #if not isinstance(image, np.ndarray):
                #    image = np.array(image)
                
                image = self.image_processing(image)
                images.append(image)
            return torch.stack(images).to(self.device) 
        except Exception as e:
            logger.error(f"Preprocessing Error: {str(e)}")
            logger.error(e, exc_info=True)

    def postprocess(self, data):
        try:
            probs, classes = torch.topk(data, dim=1, k=1)
            probs = probs.tolist()
            classes = classes.tolist()
            return map_class_to_label(probs=probs, mapping=self.mapping, lbl_classes=classes)
        except Exception as e:
            logger.error(f"Preprocessing Error: {str(e)}")
            logger.error(e, exc_info=True)

    def initialize(self, context):
        self.topk = 0.5
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        logger.info(f"Model Directory: {model_dir}")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        logger.info(f"Model Path: {model_pt_path}")
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")
        try:
            self.model = torch.jit.load(model_pt_path)
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.error(e, exc_info=True)
        # load class mapping for classifiers
        mapping_file_path = os.path.join(model_dir, "index_to_name.json")
        self.mapping = load_label_mapping(mapping_file_path)


        self.initialized = True

    def inference(self, data, *args, **kwargs):
        result = self.model.forward(data)
        logger.info(f"{result=}")
        return result
    
    def handle(self, data, context):
        try:
            model_input = self.preprocess(data)
            output = self.inference(model_input)
            return self.postprocess(output)
        except Exception as e:
            logger.error(e, exc_info=True)
