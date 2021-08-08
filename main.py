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
from torch.autograd import Variable
import helper_functions as func
from model import vgg19

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#Parameters

alpha = 1e5
#alpha = 0
beta = 3e4
#beta = 0
num_of_steps = 500


learning_rate = 10

#Device
device = torch.device("cuda" if cuda.is_available() else "cpu")

#Model

#model = models.vgg19(pretrained=True).to(device).features
model = vgg19().to(device)

def calculate_content_loss(model, image1, image2):
    feature1 = func.extract_content_feature(model, image1).squeeze(0)
    feature2 = func.extract_content_feature(model, image2).squeeze(0)
    #return torch.sum((feature1 - feature2)**2) / 2
    return torch.nn.MSELoss(reduction='mean')(feature1, feature2)

def calculate_style_loss(model, image1, image2):
    features1 = func.extract_style_features(model, image1)
    features2 = func.extract_style_features(model, image2)
    gram_matrices1 = []
    gram_matrices2 = []
    style_loss = 0
    for i in range(5):
        gram_matrices1.append(func.gram_matrix(features1[i]))
        gram_matrices2.append(func.gram_matrix(features2[i]))

    for i in range(5):
        #temp_loss = torch.sum((gram_matrices1[i] - gram_matrices2[i]) ** 2)
        #temp_loss /= (4 * features1[i].shape[1]**2 * features1[i].shape[2]**2)
        #style_loss += 0.2 * temp_loss
        temp_loss = 0
        temp_loss += torch.nn.MSELoss(reduction='mean')(gram_matrices1[i], gram_matrices2[i])
        temp_loss /= 4 * features1[i].shape[1] ** 2 * features1[i].shape[2] ** 2
        style_loss += 0.2 * temp_loss
    style_loss /= len(features1[0])
    return style_loss

def total_loss(model, img_content, img_style, generated_image, alfa, beta):
    return alfa * calculate_content_loss(model, img_content, generated_image) + beta * calculate_style_loss(model, img_style, generated_image)


def train_model(model, num_of_steps, img_content, img_style, generated_image, alfa, beta, learning_rate):
    optimizer = optim.Adam([generated_image], learning_rate)
    for i in range(num_of_steps):
        loss = total_loss(model, img_content, img_style, generated_image, alfa, beta)
        loss.backward()
        optimizer.step()
        print(loss)
        print(i)
        optimizer.zero_grad()
        if(i % 500 == 0):
            func.show_image(generated_image)
    func.show_image(generated_image)


if __name__ == "__main__":
    content_image = func.load_image("data\\bukva.jpeg").to(device)
    style_image = func.load_image("data\\vrisak.jpg").to(device)

    #noise_img = np.random.normal(loc=0, scale=90.,size=content_image.shape).astype(np.float32)
    #noise_img = torch.from_numpy(noise_img).float()

    generated_image = func.load_image("data\\bukva.jpeg")
    generated_image = Variable(generated_image, requires_grad=True).to(device).detach().requires_grad_(True)

    train_model(model,  num_of_steps, content_image, style_image, generated_image, alpha, beta, learning_rate)
