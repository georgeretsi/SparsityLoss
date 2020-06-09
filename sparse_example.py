'''
created on Dec 28, 2019
@author: georgeretsi
'''

import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.models as models

from scipy.special import erfinv
import math

from sparse_utils import *

from wide_resnet import WideResNet

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')

parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'], help='choose dataset (cifar10 or cifar100)')

parser.add_argument('-s', '--sparsity', default='adaptive', choices=['fixed', 'adaptive'],
                    help='choose from: fixed, adaptive (deafult: adaptive)')
parser.add_argument('--pthres', type=int, default=1000, help='number of minimum required parameters for sparsification')

parser.add_argument('--starget', type=float, default=0.5)
parser.add_argument('--lv', type=float, default=10.0)

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0, metavar='G',
                    help='gpu id')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
gpu_id = args.gpu

# typical augmentation for CIFAR
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        nclasses = 10
elif args.dataset == 'cifar100':
        trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        nclasses = 100

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)

model = WideResNet(16, 8, .0, nclasses)

# enable sparsification for Conv and Linear layers above pthres. Initial pruning threshold is set lower than the requested. 
iter_sparsify(model, erfinv(.6 * args.starget) * math.sqrt(2), True, args.pthres)

# not adaptive, per-layer fixed, sparsity:
#iter_sparsify(model, erfinv(1-args.starget) * math.sqrt(2), False)

if args.cuda:
    model.cuda(gpu_id)

parameters, sparameters = [], []
for name, p in model.named_parameters():
    if ".r" in name:
        sparameters += [p]
    else:
        parameters += [p]

# ensuring convergence: slower lr on sparsity controlling pruning parameter (w/o weight decay)!
optimizer = optim.SGD([{"params":parameters}, {"params":sparameters, "lr":args.lr/100.0, "weight_decay":0}],
                      lr=args.lr, momentum=.9, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs//2, T_mult=1)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(gpu_id), target.cuda(gpu_id)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target)

        eloss = 0
        if args.sparsity == 'fixed':
            # if we want fixed sparsity per layer as a trainable procedure (ensuring smoother convergence)    
            eloss = args.lv * ((adaptive_loss(model, False, reduce=False) - 1 +  args.starget)**2).mean()
        elif args.sparsity == 'adaptive':
            eloss = args.lv * (adaptive_loss(model, False)[0] - 1 + args.starget)**2
            # if an upper bound of (parameter) sparsity is defined: 
            #eloss = args.lv * F.relu(adaptive_loss(model, False)[0] - args.starget)

        loss = loss + 1.0 * eloss

        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(gpu_id), target.cuda(gpu_id)
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\n({}) - Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


model_size = count_parameters(model)
print('number of parameters: ' + str(model_size) + ' !!!!\n')
sparsity(model, False)


for epoch in range(1, args.epochs + 1):

    train(epoch)
    scheduler.step()

    sp, nz = sparsity(model, True)
    print("overall sparsity : " + str(sp) + " with " + str(model_size-int(nz)) + " nonzero elements")

    if epoch % 1 == 0:
        test(epoch)

iter_desparsify(model)

