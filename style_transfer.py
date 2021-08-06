import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
import matplotlib.pyplot as plt
import torch.nn as nn
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def unnorm(x):
    x -= np.min(x)
    x /= np.max(x)
    x *= 255
    return x


def show_img(x):
    x = np.moveaxis(x.squeeze(0).to('cpu').detach().numpy(), 0, 2)
    x = np.uint8(unnorm(x))
    plt.imshow(x)
    plt.show()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255)),
    transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
])


def printFilters(net):
    for i, param in enumerate(net.parameters()):
        if i == 0:
            plt.imshow(param.data.numpy()[8])
            plt.show()


if __name__ == '__main__':
    with Image.open("cat.jpeg") as im:
        device = "cpu"
        img_tensor = transform(im).unsqueeze(0).to(device)
        net = models.vgg19(pretrained=True).to(device).eval()
        # contentFeatures = net(img_tensor)
        # print(np.argmax(contentFeatures[0].detach().numpy()))
        # show_img(net.features[:1][0](img_tensor))
        # print(net)





