import argparse
import os
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms as T
import torchvision.models as models


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, data_list_file, imagesize, train=True):
        with open(os.path.join(data_list_file), 'r') as f:
            imgs = f.readlines()

        imgs = [(os.path.join(root, img[:-3]), int(img[-2:].strip())) for img in imgs]
        self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        if train:
            self.transforms = T.Compose([
                T.RandomResizedCrop((imagesize, imagesize)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.Resize((imagesize, imagesize)),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        data = Image.open(sample[0])
        data = data.convert(mode="RGB")
        data = self.transforms(data)
        return data.float(), np.int(sample[1])

    def __len__(self):
        return len(self.imgs)


def main(args=None):
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--print_freq', help='Print every N batch (default 100)', type=int, default=100)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--lr_step', help='Learning rate step (default 10)', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate (default 0.1)', type=float, default=0.1)
    parser.add_argument('--weight_decay', help='Weight decay (default 0.0005)', type=float, default=0.0005)
    parser.add_argument('--optimizer', help='One of sgd, adam (default sgd)',
                        type=str, default='sgd')
    parser.add_argument('--batch_size', help='Batch size (default 20)', type=int, default=20)
    parser.add_argument('--train_list', help='Path to dataset train file list')
    parser.add_argument('--test_list', help='Path to dataset test file list')
    parser.add_argument('--dataset_root', help='Path to dataset root')
    parser.add_argument('--checkpoint_root', help='Path to checkpoint root')
    parser.add_argument('--model_name', help='Name of the model to save')

    parser = parser.parse_args(args)

    is_cuda = torch.cuda.is_available()
    logger.info('CUDA available: {}'.format(is_cuda))

    imagesize = 224
    num_workers = 0
    num_classes = 2

    model = models.resnet50(pretrained=True)
    # freeze training for all pretrained layers
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if is_cuda:
        model.cuda()
    logger.info(model)

    train_dataset = Dataset(parser.dataset_root, parser.train_list, imagesize, train=True)
    test_dataset = Dataset(parser.dataset_root, parser.test_list, imagesize, train=False)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=parser.batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=parser.batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    if is_cuda:
        criterion = criterion.cuda()

    if parser.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=model.fc.parameters(),
                                    lr=parser.lr, weight_decay=parser.weight_decay)
    elif parser.optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.fc.parameters(),
                                     lr=parser.lr, weight_decay=parser.weight_decay)
    else:
        raise ValueError('Unknown optimizer %s' % parser.optimizer)

    scheduler = StepLR(optimizer, step_size=parser.lr_step, gamma=0.1)

    logger.info('{} train iters per epoch:'.format(len(trainloader)))
    logger.info('{} test iters per epoch:'.format(len(testloader)))

    start = time.time()
    last_acc = 0.0
    for i in range(parser.epochs):
        scheduler.step()

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            if is_cuda:
                data_input = data_input.cuda()
                label = label.cuda()
            output = model(data_input)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % parser.print_freq == 0:
                speed = parser.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                logger.info('{} train epoch {} iter {} {} iters/s loss {}'.format(time_str, i, ii, speed, loss.item()))

                start = time.time()

        model.eval()
        total_count = 0
        correct_count = 0
        for ii, data in enumerate(testloader):
            data_input, label = data
            if is_cuda:
                data_input = data_input.cuda()
                label = label.cuda()

            with torch.no_grad():
                output = model(data_input)

                _, pred = torch.max(output, 1)
                correct_tensor = pred.eq(label.data.view_as(pred))
                correct_count += torch.sum(correct_tensor).cpu().numpy()
                total_count += correct_tensor.size(0)

            iters = i * len(testloader) + ii

            if iters % parser.print_freq == 0:
                speed = parser.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                logger.info('{} test epoch {} iter {} {} iters/s accuracy {}'.format(time_str, i, ii, speed,
                                                                                     correct_count / total_count))

                start = time.time()

        acc = correct_count / total_count
        logger.info('Accuracy: %f' % acc)
        if last_acc < acc:
            logger.info('Accuracy increased, saving model')
            torch.save(model.state_dict(), os.path.join(parser.checkpoint_root, parser.model_name + '_{}.pt'.format(i)))
            last_acc = acc


if __name__ == '__main__':
    main()
