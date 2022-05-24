import torch
import torchvision
from torchvision.transforms import ToTensor
import os

from scatch import meas_noise_robustness as mr

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device "{device}"')


# Load pre-trained model
# TODO download model from:
# https://torchserve.pytorch.org/mar_files/resnet-152-batch_v2.mar
# Then, unzip and move 'resnet-152-batch.pt' to this folder
model = torch.jit.load('resnet-152-batch.pt')
print('Loaded model ResNet152')
print()

script_folder_path = os.path.abspath(__file__)
dataFolder = os.path.dirname(script_folder_path) + '\\data\\'

# TODO place this file in the data folder:
# https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
imagenet_data = torchvision.datasets.ImageNet(dataFolder, split='val', 
	                                          transform=torchvision.transforms.Compose([
                                                        torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Resize((400,500))]))
data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=16, shuffle=False)
print(f'Imagenet test data loaded, batch size = {data_loader.batch_size}')
print()

# Estimate robustness
custom_sigmas = (0.25, 0.50, 1.00, 1.25)
N = 100
result = mr.meas_noise_robustness(model, data_loader,
                               MC_itr=N, alpha=0.001, sigmas=custom_sigmas, device=device)
print('Evaluation complete!')
torch.save(result, 'data/imagenet_robustness_result.pt')

# Compute proportion of images within robustness radius
rs = np.linspace(0.0, 5.0, num=1000)
Rs_vals = torch.empty((len(rs), len(custom_sigmas)))
for i, r in enumerate(rs):
    Rs_vals[i] = (result>r).sum(dim=1)/(result.size()[1])

# Graph accuracy vs radius
plt.figure()
for i, s in enumerate(custom_sigmas):
    plt.plot(rs, Rs_vals.select(dim=1, index=i), label=f'sigma = {s}')

plt.title(f'Certified Accuracy vs radius on MNIST test set (N={N})')
plt.xlabel('Radius')
plt.ylim([0.0, 1.0])
plt.ylabel('Certified Accuracy')
plt.legend()
plt.grid(True)
plt.show()

