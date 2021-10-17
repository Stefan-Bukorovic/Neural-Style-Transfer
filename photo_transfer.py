from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models
import HRnet
import helper_functions as func
import torch.cuda as cuda
import main


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',  # default style layer
                  '5': 'conv2_1',  # default style layer
                  '10': 'conv3_1',  # default style layer
                  '19': 'conv4_1',  # default style layer
                  '21': 'conv4_2',  # default content layer
                  '28': 'conv5_1'}  # default style layer
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)  # layer(x) is the feature map through the layer when the input is x
        if name in layers:
            features[layers[name]] = x

    return features


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    '''
        tensor (batch, channel, height, width)
        numpy.array (height, width, channel)
        to transform tensor to numpy, tensor.transpose(1,2,0) 
    '''
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image


def load_image(img_path, img_size=None):
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)

    transform = transforms.Compose([
        # convert the (H x W x C) PIL image in the range(0, 255) into (C x H x W) tensor in the range(0.0, 1.0)
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # this is from ImageNet dataset
    ])

    # change image's size to (b, 3, h, w)
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


def train_model(model, num_of_steps, img_content, img_style, generated_image, alfa, beta, learning_rate, optimizer, style_net):
    style_weights = {'conv1_1': 0.1,
                     'conv2_1': 0.2,
                     'conv3_1': 0.4,
                     'conv4_1': 0.8,
                     'conv5_1': 1.6}
    for i in range(num_of_steps):
        generated_image = style_net(img_content).to(device)
        target_features = get_features(generated_image, model)  # extract output image's all feature maps
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]  # output image's feature map after layer
            target_gram_matrix = func.gram_matrix(target_feature)
            style_gram_matrix = style_gram_matrixs[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram_matrix - style_gram_matrix) ** 2)
            b, c, h, w = target_feature.shape
            style_loss += layer_style_loss / (c * h * w)
        loss = alfa * content_loss +  beta * style_loss
        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()
        print(loss)
        print(i)
        optimizer.zero_grad()
        if(i % 500 == 0):
            #func.5show_image(generated_image)
            plt.imshow(im_convert(generated_image))
            plt.show()

    #func.show_image(generated_image)
    plt.imshow(im_convert(generated_image))
    plt.show()


device = torch.device("cuda" if cuda.is_available() else "cpu")

VGG = models.vgg19(pretrained=True).features
#VGG = models.vgg19(pretrained=True).eval()
VGG.to(device)
#print(VGG)
# only use VGG19 to extract features, we don't need to change it's parameters
for parameter in VGG.parameters():
    parameter.requires_grad_(False)

style_net = HRnet.HRNet()
style_net.to(device)
#print(style_net)

content_image = load_image("data\content_9.png", 400).to(device)
style_image = load_image("data\style_8.png").to(device)

alfa = 150
beta = 1

content_features = get_features(content_image, VGG)
style_features = get_features(style_image, VGG)
style_gram_matrixs = {layer: func.gram_matrix(style_features[layer]) for layer in style_features}
generated_image = content_image.clone().requires_grad_(True).to(device)
optimizer = optim.Adam(style_net.parameters(), lr=5e-3)
train_model(VGG, 2000,content_image, style_image, generated_image, alfa, beta, 5e-3, optimizer, style_net)
#style_gram_matrixs = {layer: func.gram_matrix(style_features[layer]) for layer in style_features}

#target = content_image.clone().requires_grad_(True).to(device)

