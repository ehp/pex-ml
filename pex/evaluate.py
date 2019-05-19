import argparse
import logging
import sys

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms as T


def tranform_image(filename, imagesize):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        transforms = T.Compose([
                T.Resize((imagesize, imagesize)),
                T.ToTensor(),
                normalize
        ])

        data = Image.open(filename)
        data = data.convert(mode="RGB")
        data = transforms(data)
        return data.float()


def main(args=None):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--model', help='Model filename')
    parser.add_argument('--image', help='Image filename')

    parser = parser.parse_args(args)

    is_cuda = torch.cuda.is_available()

    imagesize = 224
    num_classes = 2

    data = torch.unsqueeze(tranform_image(parser.image, imagesize), dim=0)
    if is_cuda:
        data = data.cuda()

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(parser.model))
    if is_cuda:
        model.cuda()

    model.eval()
    with torch.no_grad():
        output = model(data)
        _, pred = torch.max(output, 1)
        if pred[0] == 1:
            logger.info('Indoor image')
        else:
            logger.info('Outdoor image')

if __name__ == '__main__':
    main()
