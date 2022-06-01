from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loading import DogsCatsDataset
from denoiser import DnCNNModel


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"device is {device}")

def loss_func(predictions, target, num_x_channels):
    N = predictions.size()[0]
    norm_sq = (torch.linalg.matrix_norm(predictions - target, ord='fro')) ** 2
    flatten = torch.sum(norm_sq, -1)
    return torch.sum(flatten) / (2 * N * num_x_channels)


# Hyperparams ####################
D = 17
batch_size = 30
total_epochs = 50
learn_rate = 0.1
scaled_image_h = 180
scaled_image_w = 180
sigma = 15


# Load Data ######################
dataset = DogsCatsDataset(scaled_image_h, scaled_image_w)
c = dataset.get_num_channels()
#dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#dataiter = iter(dataloader)


# Instantiate Model ##############
model = DnCNNModel(dataset.get_num_channels(), D)
optim = torch.optim.Adam(model.parameters(), lr=learn_rate)

# Training Loop ##################
seen = set()
tot_loss = 0
base_img_counter = 0

#while len(seen) < total_epochs:
while len(seen) < total_epochs:
    rand_idx = randint(0, len(dataset) - 1)
    if rand_idx in seen:
        continue

    # select base image
    base_image, label = dataset[rand_idx]
    print(f"image index selected: {rand_idx}")

    # duplicate
    X = base_image.repeat(batch_size, 1, 1, 1)

    # noise
    V = torch.randn(X.size()) * sigma
    Y = X + V

    optim.zero_grad()
    R = model(Y)

    loss = loss_func(R, V, c)
    loss.backward()
    optim.step()

    base_img_loss = loss.item()
    print(f"loss for this base image: {base_img_loss}")
    tot_loss += base_img_loss
    print(f"total running loss: {tot_loss}")
    base_img_counter += 1
    print(f"completed {base_img_counter} base images")
    print("-----------------")

    seen.add(rand_idx)



