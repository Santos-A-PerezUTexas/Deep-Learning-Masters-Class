from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        self.transform = transform
        rgb_mean = (0.4914, 0.4822, 0.4465)
        rgb_std = (0.2023, 0.1994, 0.2010)

        transform_train = transforms.Compose([
          #transforms.RandomCrop(32, padding=4),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(rgb_mean, rgb_std),
          ])

        import csv
        from os import path
        self.data = []
        to_tensor = transforms.ToTensor()
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label, _ in reader:            #READ THREE COLUMNS AT A TIME (as opposed to row[0], row[1], etc.
                if label in LABEL_NAMES:              #this exlcudes the first line
                    image = Image.open(path.join(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)
                    if self.transform:
                      image = self.transform(image)
                    image = transform_train(image)
                    #self.data.append((to_tensor(image), label_id))
                    self.data.append((image, label_id))

    def __len__(self):
        """
        Your code here
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        return self.data[idx]

def load_data(dataset_path, num_workers=0, batch_size=128, transform=None):
    dataset = SuperTuxDataset(dataset_path, transform)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
