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

#normalization_mean = [0.485, 0.456, 0.406]
#normalization_std = [0.229, 0.224, 0.225]

device = torch.device("cuda" if cuda.is_available() else "cpu")

#Set of data transforms on loaded image

data_transforms = transforms.Compose([
        #transforms.Resize([512, 512]),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        #transforms.Normalize(mean=normalization_mean, std=normalization_std)
    ])

def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
    tensor = Normalize(rgb2bgr(Loader(image) * 255)).unsqueeze(0)
    return tensor


#  Undo the above preprocessing.
def deprocess(output_tensor):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 255
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image

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
    #x = np.uint8(x)
    #plt.imshow(x)
    #trans = transforms.ToPILImage()
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