from PIL import Image
from torch.utils.data import Dataset, DataLoader
from . import dense_transforms

#This is HW3, in here for testing ONLY.  Nov 21, 2021

class DenseSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor()):
        from glob import glob
        from os import path
        self.files = []
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        lbl = Image.open(b + '_seg.png')
        if self.transform is not None:
            im, lbl = self.transform(im, lbl)
        return im, lbl


#BEGIN DETECTION CLASS FOR HW4, ABOVE WAS FOR HW3 for Test purposes

class DetectionSuperTuxDataset(Dataset):
    def __init__(self, dataset_path, transform=dense_transforms.ToTensor(), min_size=20):
        from glob import glob
        from os import path
        self.files = []
        
        for im_f in glob(path.join(dataset_path, '*_im.jpg')):
            #print ("IN LOOP")
            #print (im_f)
            self.files.append(im_f.replace('_im.jpg', ''))
        self.transform = transform
        self.min_size = min_size
        #print("IN INIT OF DETECTION CLASS, this is the size of self.files")
        #print (len(self.files))
        #print("IN INIT OF DETECTION CLASS, this is the shape of self.files")
        #print (self.files)

    def _filter(self, boxes):
        if len(boxes) == 0:
            return boxes
        return boxes[abs(boxes[:, 3] - boxes[:, 1]) * abs(boxes[:, 2] - boxes[:, 0]) >= self.min_size]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import numpy as np
        #print (f'Utils.PY:    DetectionSuperTuxDataset-->Getitem() was just called....... for index {idx}, Oct 30 2021')
        b = self.files[idx]
        im = Image.open(b + '_im.jpg')
        nfo = np.load(b + '_boxes.npz')
        #print(f'From get_item, this is KARTS array from NPZ file, item {idx} in nfo variable:')
        #print(nfo['karts'])
        #print("From get_item, this is BOMBS array from NPZ file in nfo variable:")
        #print(nfo['bombs'])
        #print("From get_item, this is PICKUPS array from NPZ file in nfo variable:")
        #print(nfo['pickup'])


        data = im, self._filter(nfo['karts']), self._filter(nfo['bombs']), self._filter(nfo['pickup'])
        
        #self._filter(nfo['karts']), etc, are the DETECTIONS.. we are LEARNING THESE.
        #ToTheatmaps converts ALL THREE OF THESE DETECTIONS  to peak and size tensors.
        #Per the assignment:  
        """
        The final step of your detector extracts local maxima from each predicted heatmap. Each local maxima corresponds to a positive detection.
        The function detect() returns a tuple of detections as a list of five numbers per class (i.e., tuple of three lists): The confidence of the 
        detection (float, higher means more confident, see evaluation), the x and y location of the object center (center of the predicted bounding box), 
        and the ***size** of the bounding box (width and height). Your detection function may return up to 100 detections per image, each detection comes with a confidence. 
        You’ll pay a higher price for getting a high confidence detection wrong. The value of the heatmap at the local maxima (peak score) is a good confidence measure.
        Use the extract_peak function to find detected objects.
        """
        
        if self.transform is not None:
            data = self.transform(*data)
        return data


def load_detection_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DetectionSuperTuxDataset(dataset_path, **kwargs)
    print("LOADED DATASET SANTOS, this one below:")
    print(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def load_dense_data(dataset_path, num_workers=0, batch_size=32, **kwargs):
    dataset = DenseSuperTuxDataset(dataset_path, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


if __name__ == '__main__':
    dataset = DetectionSuperTuxDataset('dense_data/train')
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[100+i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='r', lw=2))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='g', lw=2))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], fc='none', ec='b', lw=2))
        ax.axis('off')
    dataset = DetectionSuperTuxDataset('dense_data/train',
                                       transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(0),
                                                                           dense_transforms.ToTensor()]))
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):

        im, *dets = dataset[100+i]
        hm, size = dense_transforms.detections_to_heatmap(dets, im.shape[1:])
        ax.imshow(F.to_pil_image(im), interpolation=None)
        hm = hm.numpy().transpose([1, 2, 0])
        alpha = 0.25*hm.max(axis=2) + 0.75
        r = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
        g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
        b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
        ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
        ax.axis('off')
    fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
