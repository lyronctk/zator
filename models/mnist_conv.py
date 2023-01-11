from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import json

DIMS = 4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(2, 2, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(2, 2, 3, 1, padding=1)
        self.fc1 = nn.Linear(DIMS*DIMS*2, 10)

    def forward(self, x):
        return F.log_softmax(self.presoftmax(x), dim=1)

    def presoftmax(self, x):
        # first conv will not be layered...
        x = self.conv1(x)
        x = F.relu(x)
        x = torch.floor(x)

        # second conv will be layered
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.floor(x)

        # third conv will be layered
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.floor(x)

        # this will be also saved
        x = x.view(-1, DIMS*DIMS*2) # 64 x 3136/2
        # print(x.shape)
        x = self.fc1(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Resize((4, 4))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    torch.save(model.state_dict(), "mnist_cnn.pt")

    print(model.state_dict().keys())

    # get first test image
    X1 = next(iter(test_loader))[0][:1]

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.conv3.register_forward_hook(get_activation('conv3'))

    print(model.state_dict()['conv2.weight'].numpy())
    print(model.state_dict()['conv2.weight'].numpy().shape)

    y1 = model.presoftmax(X1).detach()

    X1 = X1.reshape(DIMS, DIMS, 1)
    import pdb; pdb.set_trace()
    # export weights to json
    with open('json/inp1_two_conv_mnist.json', 'w') as json_file:
        in_json = {
            "x": X1.numpy().astype(int).tolist(),
            "head": {
                "W": (model.state_dict()['conv1.weight'].numpy()).round().astype(int).T.tolist(),
                "b": (model.state_dict()['conv1.bias'].numpy()).round().astype(int).tolist(),
                "a": activation['conv1'].numpy().astype(int).tolist()[0]
            },
            "backbone": [
                {
                    "W": (model.state_dict()['conv2.weight'].numpy()*(10**9)).round().astype(int).T.tolist(),
                    "b": (model.state_dict()['conv2.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                    "a": activation['conv2'].numpy().astype(int).tolist()[0]
                },
                {
                    "W": (model.state_dict()['conv3.weight'].numpy()*(10**9)).round().astype(int).T.tolist(),
                    "b": (model.state_dict()['conv3.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                    "a": activation['conv3'].numpy().astype(int).tolist()[0]
                }
            ],
            "tail": {
                "W": (model.state_dict()['fc1.weight'].numpy()*(10**9)).round().astype(int).tolist(),
                "b": (model.state_dict()['fc1.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                "a": (y1.numpy()*(10**9)).astype(int).flatten().tolist()
            },
            "scale": 10**-9,
            "label": int(y1.argmax())
        }
        json.dump(in_json, json_file)

    activation = {}

    # get second test image
    X2 = next(iter(test_loader))[0][:1]

    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv2.register_forward_hook(get_activation('conv2'))
    model.conv3.register_forward_hook(get_activation('conv3'))

    y2 = model.presoftmax(X2).detach()

    X2 = X2.reshape(DIMS, DIMS, 1)

    with open('json/inp2_two_conv_mnist.json', 'w') as json_file:
        in_json = {
            "x": X2.numpy().astype(int).flatten().tolist(),
            "head": {
                "W": (model.state_dict()['conv1.weight'].numpy()*(10**9)).round().astype(int).tolist(),
                "b": (model.state_dict()['conv1.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                "a": activation['conv1'].numpy().astype(int).tolist()[0]
            },
            "backbone": [
                {
                    "W": (model.state_dict()['conv2.weight'].numpy()*(10**9)).round().astype(int).tolist(),
                    "b": (model.state_dict()['conv2.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                    "a": activation['conv2'].numpy().astype(int).tolist()[0]
                },
                {
                    "W": (model.state_dict()['conv3.weight'].numpy()*(10**9)).round().astype(int).tolist(),
                    "b": (model.state_dict()['conv3.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                    "a": activation['conv3'].numpy().astype(int).tolist()[0]
                }
            ],
            "tail": {
                "W": (model.state_dict()['fc1.weight'].numpy()*(10**9)).round().astype(int).tolist(),
                "b": (model.state_dict()['fc1.bias'].numpy()*(10**9)).round().astype(int).tolist(),
                "a": (y2.numpy()*(10**9)).astype(int).flatten().tolist()
            },
            "scale": 10**-9,
            "label": int(y2.argmax())
        }
        json.dump(in_json, json_file)

    # with open('json/784_single_dense.json', 'w') as json_file:
    #     in_json={
    #         "x": X2.numpy().astype(int).flatten().tolist(),
    #         "weights": (model.state_dict()['fc1.weight'].numpy()*(10**6)).round().astype(int).tolist(),
    #         "bias": (model.state_dict()['fc1.bias'].numpy()*(10**6)).round().astype(int).tolist(),
    #     }

    #     json.dump(in_json, json_file)


if __name__ == '__main__':
    main()
