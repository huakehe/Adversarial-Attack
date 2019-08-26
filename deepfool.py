from __future__ import print_function
import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import math
import torchvision.models as models
from PIL import Image
import os

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
        return x


print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


model = Net().to(device)
para = torch.load('trained.pth.tar',map_location='cpu')
model.load_state_dict(para)
model.eval()


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=4):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()

    f_image = net.forward(Variable(image, requires_grad=True)).data.cpu().numpy().flatten()
  
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image, requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    #return r_tot, loop_i, label, k_i, pert_image
    return pert_image


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            ])), 
        batch_size=1, shuffle=True)



def test( model, device, dt, tar, net):
    correct = 0
    attack_succeed = 0
    adv_examples = []
    org_examples = []

    for z in range(0,1000):
        data = dt[z].to(device)
        org_img = data.clone()
        org = org_img.squeeze().detach().cpu().numpy()
        target = tar[z].to(device)
        data.requires_grad = True
        pert_image = deepfool(data , model)

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] 
        if init_pred.item() != target.item():
            continue

        output = model(pert_image)
        final_pred = output.max(1, keepdim=True)[1]
        
        if final_pred == target.item():
            correct += 1
        else:
            if len(adv_examples) < 5:
                attack_succeed += 1
                adv_ex = pert_image.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
                org_examples.append(org)
    ASR = attack_succeed/(correct+attack_succeed)
    return ASR, adv_examples, org_examples



dt = torch.load('1000_data.pth.tar')
tar = torch.load('1000_tar.pth.tar')
asr,examples,org_examples = test(model, device, dt, tar, model)
print("adv attack success rate is: {:4.2f}%".format(asr*100))


fig = plt.figure(figsize=(8,10))
len_ex = len(examples)
count = 1
for j in range(len(examples)):
    org_ex = org_examples[j]
    orig,adv,ex = examples[j]
    a = fig.add_subplot(len_ex, 2, count)
    imgplot = plt.imshow(org_ex,cmap="gray")
    a.set_title('original -> {}'.format(orig))
    count += 1
    a = fig.add_subplot(len_ex, 2, count)
    imgplot = plt.imshow(ex,cmap="gray")
    a.set_title('adv -> {}'.format(adv))
    count += 1
plt.tight_layout()
plt.show()
plt.savefig('df.png')
