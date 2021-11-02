import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary

from .models import Detector, CNNClassifier, save_model
from .utils import load_detection_data, load_dense_data
from . import dense_transforms
import torch.utils.tensorboard as tb
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]


class FocalLoss(nn.Module):
    
    def __init__(self, weight=None, 
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input_tensor, target_tensor):

        print ("IN forward of Focal Loss now")
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        
        new_input_tensor = ((1 - prob) ** self.gamma) * log_prob
        print (f'Shape of new_input_tensor is {new_input_tensor.shape}')
        #print (f'Shape of weights is {self.weight.shape}')
        print (f'Shape of target_tensor is {target_tensor.shape}')
        loss = F.nll_loss(new_input_tensor,  target_tensor, weight=self.weight, reduction = self.reduction).to(input_tensor.device) 

        print (f'the shape of loss in focalloss() is {loss.shape}')

        return loss










def train(args):
    
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector().to(device)
    model2 = CNNClassifier().to(device)
    #print(model)
    #summary(model, (3,96,128))

    train_logger, valid_logger = None, None
    
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    w=w.to(device)
    loss = torch.nn.CrossEntropyLoss(weight=w / w.mean()).to(device)
    #focal_loss = FocalLoss(weight=w / w.mean()).to(device).to(device)
    focal_loss = FocalLoss().to(device).to(device)
    #NOTE FOCAL LOSS HAS NO WEIGHTS!!
    #NOTE FOCAL LOSS HAS NO WEIGHTS!!
    #NOTE FOCAL LOSS HAS NO WEIGHTS!!
    #NOTE FOCAL LOSS HAS NO WEIGHTS!!
    
    print ("1................FROM TRAIN() ABOUT TO LOAD DATA")
    #val = input("PRESS ANY KEY")
    #print(val)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)

    print (f'2................Size of training set is {len(train_data)}')
    

    global_step = 0
    for epoch in range(1):   #WARNING CHANGE TO args.num_epoch   #WARNING CHANGE TO args.num_epoch

        print(f'***********At the beggining of epoch  {epoch+1}****************')
        model.train()
        
        #batch size is 32
        #batch size is 32

        i_pred = 0
        batch = 0

        for img, peaks, size in train_data:        #THIS CALLS GET ITEM 145 TIMES OCT 30 2021
            
            img, peaks, size  = img.to(device), peaks.to(device).long(),  size.to(device).long()
            #NOV 21, 2021:  img, peaks, size  = img.to(device), peaks.to(device),  size.to(device)           
            #ToTheatmaps converts ALL THREE OF THESE DETECTIONS  to peak and size tensors.
            #Per the assignment:  
            """
            The final step of your detector extracts local maxima from each predicted heatmap. Each local maxima corresponds to a positive detection.
            The function detect() returns a tuple of detections as a list of five numbers per class (i.e., **tuple of three lists): The confidence of the 
            detection (float, higher means more confident, see evaluation), the x and y location of the object center (center of the predicted bounding box), 
            and the ***size** of the bounding box (width and height). Your detection function may return up to 100 detections per image, each detection comes with a confidence. 
            You’ll pay a higher price for getting a high confidence detection wrong. The value of the heatmap at the local maxima (peak score) is a good confidence measure.
            Use the extract_peak function to find detected objects.
            """
        
            print (f'            (IN LOOOP NOW)....-------------This is batch number {batch+1}, of size {len(img)}')
            batch +=1    
            

            print (f'TRAIN() ----->MAKING PREDICTION NUMBER {i_pred+1} WITH detected_peaks=model(img)-----------')
            print (f'            Image shape is {img.shape}')
            #Image shape is torch.Size([32, 3, 96, 128])
            print (f'             peaks shape is {peaks.shape}')
            #peaks shape is torch.Size([32, 3, 96, 128])
            print (f'              size shape is {size.shape}')
            #size shape is torch.Size([32, 2, 96, 128])            
            
            i_pred += 1

            detected_peaks = model(img)
            #labels = model2(img)

            
            print (f'              (LOOP)peaks shape is {peaks.shape}')
            print (f'              (LOOP) DETECTED peaks shape is {detected_peaks.shape}')
            print("                   (LOOP)  GOING TO COMPUTE THE LOSS NOW")
            
            #loss_val = loss(detected_peaks, peaks)

            print (f'Detected peaks shape is {detected_peaks.shape}') #([32, 3, 96, 128])
            reduced_peaks = peaks[:, 0, :, :]
            print (f'reduced peaks dimension is {reduced_peaks.shape} ') #[32, 96, 128]

            focal_Loss= focal_loss(detected_peaks, reduced_peaks)   #peaks or reduced_peaks
            
            #Nov 2, 2021:  Output dimesion is [32, 3, 96, 128], the label dimension should be [32, 96, 128], long type integers.
            #https://piazza.com/class/ksjhagmd59d6sg?cid=776

            print (f'              (LOOP)LOSS  shape is {focal_Loss.shape}')
                        
            print (f'  (LOOP)   Finished making prediction/detection for batch {batch} ')

          
        save_model(model)
        print("DONE")

  

#--------------------------------------END TRAIN


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-g', '--gamma', type=float, default=0, help="class dependent weight for cross entropy")
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap()])')
    #parser.add_argument('-t', '--transform',
     #                   default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)


    """
      if train_logger is not None and global_step % 100 == 0:
                log(train_logger, img, label, logit, global_step)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            conf.add(logit.argmax(1), label)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('global_accuracy', conf.global_accuracy, global_step)
            train_logger.add_scalar('average_accuracy', conf.average_accuracy, global_step)
            train_logger.add_scalar('iou', conf.iou, global_step)

        model.eval()
        val_conf = ConfusionMatrix()
        for img, label in valid_data:
            img, label = img.to(device), label.to(device).long()
            logit = model(img)
            val_conf.add(logit.argmax(1), label)

        if valid_logger is not None:
            log(valid_logger, img, label, logit, global_step)

        if valid_logger:
            valid_logger.add_scalar('global_accuracy', val_conf.global_accuracy, global_step)
            valid_logger.add_scalar('average_accuracy', val_conf.average_accuracy, global_step)
            valid_logger.add_scalar('iou', val_conf.iou, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f \t iou = %0.3f \t val iou = %0.3f' %
                  (epoch, conf.global_accuracy, val_conf.global_accuracy, conf.iou, val_conf.iou))
    """
