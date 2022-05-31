import numpy as np
import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms as T
from PIL import Image

from denoiser_training import DnCNN

data_dir = 'data/dogs_cats/train/'
filename = 'cat.7744.jpg'
demo_img = io.read_image(data_dir + filename)

model = torch.load('denoiser_models/DnCNN_sigma25/model_002.pth')

# if just using on some images for demo purposes, move to cpu:
model = model.cpu()

sigma = 25
# make sure the noise is scaled to the [0, 1] range
noise = torch.randn(demo_img.size()) * sigma / 255.0

# put the demo image (tensor) in range [0, 1], then add the noise
y = demo_img / 255.0 + noise

# transform the noisy demo tensor back to an image so we can save it and look at it
transform = T.ToPILImage()
y_img = transform(y)
y_img.save("noisy_color_cat.jpg")

# the model expects the tensor to include a dimension for the batch
y = torch.unsqueeze(y, 0)
denoised_img = model(y)
denoised_img = transform(torch.squeeze(denoised_img))
denoised_img.save("denoised_color_cat.jpg")
