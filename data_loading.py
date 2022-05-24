from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize


# to make a DataLoader, do the following:

# from torch.utils.data import DataLoader
# dataset = DogsCatsDataset()
# dataloader = DataLoader(dataset=dataset, batch_size=some_num, shuffle=True)
# data_iter = iter(dataloader) ... or ... enumerate(dataloader)


class DogsCatsDataset(Dataset):
    """
    Creates the dogs_cats training dataset. Must have the 'dc_train_filenames.txt' file
    in the 'data/dogs_cats/' directory.
    Assumes that the images are in the directory: 'data/dogs_cats/train/'
    Resizes all images to a standard size given as height, width parameters.
    Defaults to 200x200.

    A cat image has a label of 0.
    A dog image has a label of 1.
    """

    def __init__(self, height=200, width=200):
        self.directory = "data/dogs_cats/"
        self.h = height
        self.w = width
        self.data_keys_list = []
        self.labels = []
        self.map_fn_2_idx = {}
        self.map_idx_2_fn = {}

        with open(self.directory + "dc_train_filenames.txt", "r") as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                filename = line.strip()
                self.map_fn_2_idx[filename] = i
                self.map_idx_2_fn[i] = filename
                self.data_keys_list.append(filename)
                if filename[:3] == "cat":
                    self.labels.append(0)
                else:
                    self.labels.append(1)

        self.num_samples = len(self.data_keys_list)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        path = self.directory + "train/" + self.data_keys_list[index]
        img = read_image(path)
        return resize(img, [self.h, self.w]), self.labels[index]

    def get_filename(self, index):
        return self.map_idx_2_fn[index]

    def get_index(self, filename):
        return self.map_fn_2_idx[filename]
