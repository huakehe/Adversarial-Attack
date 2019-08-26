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

#epsilons = [0.1]
epsilons = [0,.05,0.1,0.15,0.2,0.25,0.3]
l = [] # list of adv success rate
use_cuda=True

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

#print(type(test_loader))

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net().to(device)

# CHANGED: Load my trained model
para = torch.load('trained.pth.tar',map_location='cpu')
model.load_state_dict(para)

#model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()

def untarget_fgsm_attack(image, epsilon, data_grad, step):

    # step is T, alpha = epsilon/T
    alpha = epsilon / step
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + alpha * sign_data_grad
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
        if(count[num]<100):
            count[num] = count[num] + 1
            reduced_data.append(data)
            reduced_tar.append(target)
    
    torch.save(reduced_data,"1000_data.pth.tar")
    torch.save(reduced_tar,"1000_tar.pth.tar")
    

def test( model, device, dt, tar, epsilon, step ):
    st = step
    correct = 0
    for z in range(0,1000):
        #data = dt[z].to(device)
        org_data = dt[z].to(device)
        data = org_data.clone()
        target = tar[z].to(device)
        step = st
        while(step>0):
            # make data a autograd variable
            data = torch.autograd.Variable(data, requires_grad=True)
            output = model(data)
            loss = F.nll_loss(output, target)
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data
            perturbed_data = untarget_fgsm_attack(data, epsilon, data_grad, step)
            data = perturbed_data.clone()
            step = step - 1
            
        output = model(data) 
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1

    final_acc = correct/float(1000)
    l.append((1-final_acc)*100)
    print("Epsilon: {}\t accuracy = {} / {} = {}".format(epsilon, correct, 1000, final_acc))
    return final_acc


#### main

reduce(test_loader)
dt = torch.load('1000_data.pth.tar')
tar = torch.load('1000_tar.pth.tar')
step = 5


accuracies = []
examples = []

# Run test for each epsilon
for eps in epsilons:
    acc= test(model, device, dt, tar, eps, step)
    accuracies.append(acc)

print("EPS | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |")
print("ASR |{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|".format(l[0],l[1],l[2],l[3],l[4],l[5],l[6]))


