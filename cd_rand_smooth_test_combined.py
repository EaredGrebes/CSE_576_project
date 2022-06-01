import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
import numpy as np
import os

import classifier_models as cm
from scatch import meas_noise_robustness as mr

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device "{device}"')

class CatsDogsDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.flist = os.listdir(img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.classes=('dog', 'cat')

    def __len__(self):
        return len(self.flist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.flist[idx])
        image = torchvision.io.read_image(img_path)
        label = int(self.flist[idx][0:3]==self.classes[1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def main():
    # Data can be downloaded from Kaggle: 
    # https://www.kaggle.com/competitions/dogs-vs-cats/data
    cats_and_dogs_data_dir = './data/dogs_cats/train'
    sized_img_width        = 224 # Width and height to resize all images to
    val_set_size           =  64 # 256 # Number of images to use in validation set (chosen randomly)
    batch_size             =   1 # Images to run at once

    data_tf = tfs.Compose([tfs.Resize((sized_img_width, sized_img_width))])
    cd_data = CatsDogsDataset(cats_and_dogs_data_dir, transform=data_tf)
    cd_data_subset = torch.utils.data.random_split(cd_data, 
                                                   [val_set_size, len(cd_data)-val_set_size], 
                                                   generator=torch.Generator().manual_seed(42))[0]
    data_loader = DataLoader(cd_data_subset, batch_size=batch_size, shuffle=False)
    print(f'Cats-and-dogs data loaded. ({len(cd_data_subset)} images only)')
    # for d, l in data_loader:
    #     print(f'batch 0 labels are: {l}')
    #     idx = 6
    #     plt.imshow(d[idx].sum(dim=0), cmap='gray')
    #     plt.title(f'Class = {l[idx}')
    #     plt.show()
    #     break

    # Load pre-trained model
    model = cm.binaryResNet()
    model.load_state_dict(torch.load('cats_dogs_resNet18.pt', map_location=device))
    print('Loaded model binaryResNet.')

    # Estimate robustness
    custom_sigmas = (3.0, 6.0, 9.0)
    N = 200
    result = mr.meas_noise_robustness(model.to(device=device), data_loader,
                                      MC_itr=N, alpha=0.001, sigmas=custom_sigmas, 
                                      device=device)
    print('Evaluation complete!')
    torch.save(result, 'data/imagenet_robustness_result.pt')

    # Compute proportion of images within robustness radius
    rs = np.linspace(0.0, 8.0, num=1000)
    Rs_vals = torch.empty((len(rs), len(custom_sigmas)))
    for i, r in enumerate(rs):
        Rs_vals[i] = (result>r).sum(dim=1)/(result.size()[1])

    # Graph accuracy vs radius
    plt.figure()
    for i, s in enumerate(custom_sigmas):
        plt.plot(rs, Rs_vals.select(dim=1, index=i), label=f'sigma = {s}')

    plt.title(f'Certified Accuracy vs radius on Cats/Dogs test set (N={N})')
    plt.xlabel('Radius')
    plt.ylim([0.0, 1.0])
    plt.ylabel('Certified Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
