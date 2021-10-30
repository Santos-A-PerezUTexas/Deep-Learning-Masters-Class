import torch
import numpy as np
from torchvision import models
from torchsummary import summary

from .models import Detector, save_model
from .utils import load_detection_data
from . import dense_transforms
import torch.utils.tensorboard as tb
DENSE_CLASS_DISTRIBUTION = [0.52683655, 0.02929112, 0.4352989, 0.0044619, 0.00411153]



def train(args):
    
    from os import path
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Detector()
    #print(model)
    #summary(model, (3,96,128))

    train_logger, valid_logger = None, None
    

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    loss = torch.nn.CrossEntropyLoss(weight=w / w.mean()).to(device)
     

    
    print ("FROM TRAIN() ABOUT TO LOAD DATA!!!!!!!!!!!!!!!") 
   
    print ("FROM TRAIN() ABOUT TO LOAD DATA!!!!!!!!!!!!!!!")
    #val = input("PRESS ANY KEY")
    #print(val)

    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})
    train_data = load_detection_data('dense_data/train', num_workers=4, transform=transform)
    valid_data = load_detection_data('dense_data/valid', num_workers=4)

    print ("FROM TRAIN() FINISHED LOAD DATA!!!!!!!!!!!!!!!")
    print ("FROM TRAIN() FINISHED LOAD DATA!!!!!!!!!!!!!!!")
    print ("FROM TRAIN() FINISHED LOAD DATA!!!!!!!!!!!!!!!")
    print ("FROM TRAIN() FINISHED LOAD DATA!!!!!!!!!!!!!!!")

    print (f'Size of training set is {len(train_data)}')
    #val = input("PRESS ANY KEY")
    #print(val)

    global_step = 0
    for epoch in range(args.num_epoch):

        print("At the beggining of an epoch****************")
        model.train()
        
        #batch size is 32
        #batch size is 32
        i_pred = 0

        #for img, label, size in train_data
        for img, karts, bombs, pickup in train_data:        #THIS CALLS GET ITEM 145 TIMES OCT 30 2021
            
            img, karts, bombs  = img.to(device), karts.to(device),  bombs.to(device)
            #img, label, size, pickup = img.to(device), label.to(device).long(),  size.to(device).long() 
            

            print (f'MAKING PREDICTION NUMBER {i_pred} WITH logit=model(img)-----------')
            print (f'MAKING PREDICTION NUMBER {i_pred} WITH logit=model(img)-----------')
            print (f'MAKING PREDICTION NUMBER {i_pred} WITH logit=model(img)-----------')
            print (f'Image shape is {img.shape}')
            #Image shape is torch.Size([32, 3, 96, 128])
            print (f'karts shape is {karts.shape}')
            #karts shape is torch.Size([32, 3, 96, 128])
            print (f'bombs shape is {bombs.shape}')
            #bombs shape is torch.Size([32, 2, 96, 128])
            #print (f'pickup shape is {pickup.shape}')
            #pickups shape is 


            print(f'image, label, and karts, bombs, and pickup  number 25 is IMAGE: {img[25]}, KARTS: {karts[25]}, BOMBS->:{bombs[25]},  PICKUP: ->:{11111},  respectively.')
            #print(f'THE MEAN FOR: image, label, and size number 25 is IMAGE: {img[25].mean}, LABEL: {label[25].mean}, SIZE:{size[25].mean}, respectively.')
            
            logit = model(img)
            loss_val = loss(logit, label)
            i_pred += 1
            print ("Finished making prediction/detection")

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
        save_model(model)

  

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

    args = parser.parse_args()
    train(args)
