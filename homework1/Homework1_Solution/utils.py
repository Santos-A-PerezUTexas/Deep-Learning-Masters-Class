from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        import csv
        from os import path
        self.data = []
        to_tensor = transforms.ToTensor()
        with open(path.join(dataset_path, 'labels.csv'), newline='') as f:
            reader = csv.reader(f)
            for fname, label, _ in reader:
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, fname))
                    label_id = LABEL_NAMES.index(label)
                    self.data.append((to_tensor(image), label_id))

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


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()


if __name__ == "__main__":
    """
    Optional:
    This code down here allows you to measure how fast your data loader is.
    The master solution takes about 2s to __init__ on a desktop CPU, and 0.2 / epoch.
    """
    from time import time
    import argparse
    parser = argparse.ArgumentParser("Loads the entire dataset once")
    parser.add_argument('dataset')
    parser.add_argument('-d', '--use_data_loader', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-n', '--n_epoch', type=int, default=1)
    args = parser.parse_args()

    if args.use_data_loader:
        L = load_data
    else:
        L = SuperTuxDataset

    t0 = time()
    data = L(args.dataset)

    epoch_t = [time() - t0]
    for i in range(args.n_epoch):
        list(data)
        epoch_t.append(time() - t0)
    if args.plot:
        import pylab as plt
        plt.plot(epoch_t)
        plt.show()
    else:
        print('Timing:', epoch_t)
