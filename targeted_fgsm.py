from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

epsilon = 0.3
use_cuda=True
distribution=np.zeros((10,10))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])), 
        batch_size=1, shuffle=True)


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# CHANGED: Load my trained model
para = torch.load('trained.pth.tar',map_location='cpu')
model.load_state_dict(para)

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def targeted_fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign() # this data grad is the targeted class
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image - epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def reduce(test_loader):
    count = [0]*10
    reduced_data = []
    reduced_tar = []
    for data, target in test_loader:
        num = target.data[0]
        if(count[num]<10):
            count[num] = count[num] + 1
            reduced_data.append(data)
            reduced_tar.append(target)
    torch.save(reduced_data,"100_data.pth.tar")
    torch.save(reduced_tar,"100_tar.pth.tar")

def get_tar_N(data,target,n):
    n_list = []
    for x in range(0,100):
        if(target[x]==n):
            n_list.append(x)

    return n_list


def test( model, device, data, target, epsilon, n_list ):
    correct = 0
    for z in n_list:
        org_data = dt[z].to(device)
        data = org_data.clone()
        target = tar[z].to(device)
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # original pic prediction
        if init_pred.item() != target.item():
            continue

        for u in range(0,10):
            # print(torch.sum(org_data == data))
            data = torch.autograd.Variable(data, requires_grad=True)
            output = model(data)
            new_tar = torch.tensor([u],device=torch.device('cuda')) 

            loss = F.nll_loss(output, new_tar)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            # Call targeted FGSM Attack, t is target
            perturbed_data = targeted_fgsm_attack(data, epsilon, data_grad)
            output = model(perturbed_data)

            final_pred = output.max(1, keepdim=True)[1] # perturbed pic prediction
            if(final_pred.data[0] == u):
                distribution[target.item()][final_pred.item()] += 1

reduce(test_loader)
dt = torch.load('100_data.pth.tar')
tar = torch.load('100_tard.pth.tar')

# 0-9 dataset
for v in range(0,10):
    test(model, device, dt, tar, epsilon, get_tar_N(dt,tar,v))

#for r in range(0,10):
#    distribution[r][r] = 0

print(distribution)
plt.figure(figsize=(10,10))
plt.imshow(distribution, cmap='gray_r', interpolation='none', aspect='equal')
plt.show()
plt.savefig('tar_fgsm.png')

