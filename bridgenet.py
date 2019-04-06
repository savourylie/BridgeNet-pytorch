import os

from dstorch.data import calc_data_stats, Cutout
from dstorch.utils import random_weight_init
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from tqdm import tqdm, tqdm_notebook, trange

GPU_IDS = '1, 2'
# GPU_IDS = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IDS

data_path = 'data/'
batch_size = 128

cifar10_mean_tuple, cifar10_std_tuple = (0.4914, 0.48216, 0.44653), (0.1281, 0.1242, 0.1551)
cifar100_mean_tuple, cifar10_std_tuple = (0.5070, 0.4865, 0.4408), (0.1626, 0.1539, 0.1774)

train_transformer = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(cifar10_mean_tuple, cifar10_std_tuple),
                                        Cutout(n_holes=1, length=16)
                                        ])

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(cifar10_mean_tuple, cifar10_std_tuple),

                                       ])

cifar10_training_set = datasets.CIFAR10(root=data_path, train=True, download=True, transform=train_transformer)
cifar10_test_set = datasets.CIFAR10(root=data_path, train=False, transform=test_transformer)

cifar10_train_loader = torch.utils.data.DataLoader(cifar10_training_set, batch_size=batch_size, shuffle=True)
cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_set, batch_size=batch_size, shuffle=True)

# cifar100_training_set = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transformer)
# cifar100_test_set = datasets.CIFAR100(root=data_path, train=False, transform=transformer)

# cifar100_train_loader = torch.utils.data.DataLoader(cifar100_training_set, batch_size=batch_size, shuffle=True)
# cifar100_test_loader = torch.utils.data.DataLoader(cifar100_test_set, batch_size=batch_size, shuffle=True)

class BridgeNet(nn.Module):
    def __init__(self, pretrained=False, freeze_features=False, route=None):
        super().__init__()

        self.route = route
        self.dropout_fc = 0.1

        # Feature layers
        self.feature_layernames = list(models.resnet50(pretrained=pretrained).children())[:-1]
        #         self.feature_layernames = list(models.resnet34(pretrained=pretrained).children())[:-1]
        self.feature = nn.Sequential(*self.feature_layernames)

        # CIFAR10 Classifier
        self.cifar10_layer_dict = nn.ModuleDict([
            ['fc1', nn.Linear(2048, 512)],
            ['relu1', nn.ReLU()],
            ['dp1', nn.Dropout(self.dropout_fc)],
            ['fc2', nn.Linear(512, 128)],
            ['relu2', nn.ReLU()],
            ['dp2', nn.Dropout(self.dropout_fc)],
            ['logit', nn.Linear(128, 10)],
            ['log_softmax', nn.LogSoftmax(dim=1)]
        ])

        # CIFAR100 Classifier
        self.cifar100_layer_dict = nn.ModuleDict({
            'fc1': nn.Linear(512, 256),
            'relu1': nn.ReLU(),
            'dp1': nn.Dropout(self.dropout_fc),
            'fc2': nn.Linear(256, 128),
            'relu2': nn.ReLU(),
            'dp2': nn.Dropout(self.dropout_fc),
            'logit': nn.Linear(128, 100),
            'log_softmax': nn.LogSoftmax(dim=1)
        })

        if freeze_features:
            self.freeze_feature_layers()

    def forward(self, x):
        x = self.feature(x)

        batch_size, num_channels, height, weight = x.shape
        x = x.view(-1, num_channels * height * weight)

        if self.route == 'cifar10':
            for layername, layer in self.cifar10_layer_dict.items():
                x = layer(x)
        #             for layer in self.cifar10_layer_list:
        #                 x = layer(x)

        elif self.route == 'cifar100':
            cifar10_hlayer_dict = {}

            h = x

            for layername, layer in self.cifar10_layer_dict.items():
                h = layer(h)
                cifar10_hlayer_dict[layername] = h

            for layername, layer in self.cifar100_layer_dict.items():
                if layername != 'logit' and layername != 'log_softmax':
                    x = cifar10_hlayer_dict[layername] + self.cifar10_layer_dict[layername](x)
                else:
                    x = layer(x)
        else:
            raise ValueError("Route is not set!")

        return x

    def set_route(self, route):
        self.route = route

    def set_training_type(self, ttype):
        self.ttype = ttype

    def init_weight(self, modulename):
        modulename_dict = {
            'feature': self.feature,
            'cifar10': self.cifar10_layer_dict,
            'cifar100': self.cifar100_layer_dict
        }

        random_weight_init(modulename_dict[modulename])

    def freeze_module(self, modulename):
        if modulename == 'feature':
            for child in self.feature.children():
                for param in child.parameters():
                    param.requires_grad = False

        elif modulename == 'cifar10':
            for child in self.cifar10_layer_dict.children():
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_feature_layers(self):
        if modulename == 'feature':
            for child in self.feature.children():
                for param in child.parameters():
                    param.requires_grad = True

        elif modulename == 'cifar10':
            for child in self.cifar10_layer_dict.children():
                for param in child.parameters():
                    param.requires_grad = True


from sklearn.metrics import accuracy_score


def train(model, train_loader, test_loader, loss_func, num_epochs, log_interval=100):
    loss_list, acc_list = [], []
    loss_test_list, acc_test_list = [], []
    cycle = 1

    for epoch in range(1, num_epochs + 1):
        model.train()

        # SGDR Warm start
        if epoch == cycle:
            optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.5, weight_decay=0.0001, nesterov=True)
            optim_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * cycle,
                                                                   eta_min=0, last_epoch=-1)
            cycle = cycle * 2

        for i, (data, target) in enumerate(train_loader):
            if torch.cuda.is_available():
                model = model.cuda()
                data, target = data.cuda(), target.cuda()

            data, target = Variable(data, requires_grad=True), Variable(target)

            optimizer.zero_grad()
            output = model(data)

            loss_sum = loss_func(output, target).mean()

            loss_sum.backward()
            optimizer.step()
            optim_scheduler.step()

            loss_list.append(loss_sum.detach())

            if i % log_interval == 0:
                print("Epoch: {0}, Iter: {1}, Train loss: {2:.4f}, LR: {3:.6f}".format(epoch, i, loss_sum,
                                                                                       optim_scheduler.get_lr()[0]))

        model.eval()

        test_loss = 0
        correct = 0
        num_rows = 0

        with torch.no_grad():
            for i, (data_test, target_test) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data_test, target_test = data_test.cuda(), target_test.cuda()

                data_test, target_test = Variable(data_test), Variable(target_test)
                num_rows += data_test.size(0)

                output = model(data_test)
                loss_test = loss_func(output, target_test).mean()

                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target_test.data.view_as(pred)).cpu().sum()

            test_loss /= num_rows

            print("=====================================================================")
            print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, num_rows, (100. * correct.item()) / num_rows))
            print("=====================================================================")


model = BridgeNet(pretrained=True, route='cifar10')
model.init_weight('cifar10')
model = nn.DataParallel(model)

loss_func = nn.NLLLoss()

train(model, cifar10_train_loader, cifar10_test_loader, loss_func, num_epochs=127)