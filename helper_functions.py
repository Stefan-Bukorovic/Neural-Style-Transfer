import numpy.random
import numpy as np
import torch.cuda as cuda
import torchvision.models as models
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import os



IMAGENET_MEAN_1 = [123.675, 116.28, 103.53]
IMAGENET_STD_1 = [1, 1, 1]

device = torch.device("cuda" if cuda.is_available() else "cpu")

#Set of data transforms on loaded image

data_transforms = transforms.Compose([
        #transforms.Resize([800, 600]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])

#Load image with given path, apply transforms, and unsqueeze it

def load_image(image_path):
    loaded_image = Image.open(image_path)
    image = data_transforms(loaded_image).unsqueeze(0)
    return image

#Imshow for tensor image

def unnormalize(image):
    x = image
    x -= np.min(x)
    x /= np.max(x)
    x *= 255
    return x

def show_image(image):
    x = np.moveaxis(image.squeeze(0).cpu().detach().numpy(), 0, 2)
    x = np.uint8(unnormalize(x))
    plt.imshow(x)
    plt.show()

#Calculate gram matrix
#Dimension of features is number_of_matrices x matrix_height x matrix_width
def gram_matrix(input):
    batch, channel, height, width = input.size()
    features = input.view(channel, height * width)
    gram = torch.mm(features, features.t())
    return gram

def extract_content_feature(model, image):
    #feature = model[:6](image).to(device)
    image = model(image)
    feature = getattr(image, 'conv4_2')
    return feature

def extract_style_features(model, image):
    features = []
    image = model(image)
    conv_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    for layer in conv_layers:
        features.append(getattr(image, layer))
    return features