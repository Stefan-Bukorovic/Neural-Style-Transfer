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
        transforms.Resize([256, 256]),
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
def gram_matrix(features):
    num_of_features = features.shape[0]
    gram = torch.zeros(num_of_features, num_of_features)

    for i in range(num_of_features):
        for j in range(num_of_features):
            first_vec = torch.flatten(features[i])
            second_vec = torch.flatten(features[j])
            gram[i][j] = torch.dot(first_vec, second_vec)

    return gram


def extract_content_feature(model, image):
    feature = model[:6](image).to(device)
    return feature

def extract_style_features(model, image):
    conv_ind = [0, 5, 10, 19, 28]
    features = []
    for ind in conv_ind:
        features.append(model[:ind+1](image))
    return features