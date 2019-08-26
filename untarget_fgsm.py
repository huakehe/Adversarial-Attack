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

epsilons = [0,.05,0.1,0.15,0.2,0.25,0.3]

#pretrained_model = "data/lenet_mnist_model.pth"
use_cuda=True
l = [] # list of adv success rate

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

def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
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
    '''
    print("r_data length: ",len(reduced_data))
    print("r_tar length: ",len(reduced_tar))
    '''
    torch.save(reduced_data,"1000_data.pth.tar")
    torch.save(reduced_tar,"1000_tar.pth.tar")
    

def test( model, device, dt, tar, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    count = [0]*10

    for z in range(0,1000):
        data = dt[z].to(device)
        target = tar[z].to(device)
        #print("inside list data type is: ",type(data))
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        if init_pred.item() != target.item():
            continue
        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                #print("adv ex type: ",type(adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

   
    sum = 1000
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(sum)
    l.append((1-final_acc)*100)
    print("Epsilon: {}\t accuracy = {} / {} = {}".format(epsilon, correct, sum, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

reduce(test_loader)
dt = torch.load('1000_data.pth.tar')
tar = torch.load('1000_tar.pth.tar')

accuracies = []
examples = []

for eps in epsilons:
    acc, ex = test(model, device, dt, tar, eps)
    accuracies.append(acc)
    examples.append(ex)

print("EPS | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |")
print("ASR |{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|{:4.2f}%|".format(l[0],l[1],l[2],l[3],l[4],l[5],l[6]))


plt.figure(figsize=(5,5))
plt.plot(epsilons, l, "*-")
plt.yticks(np.arange(0, 100, step=10))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("ADV Success Rate vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("ASR(%)")
plt.show()
plt.savefig('plot.png')


cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('examples.png')



plt.figure(figsize=(3,3))
ex2 = examples[3][4]
plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
plt.savefig('5.png')


