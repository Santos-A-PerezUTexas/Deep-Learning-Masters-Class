import torch
import numpy as np

from .models import FCN, save_model, ClassificationLoss
from .utils import load_dense_data, DENSE_CLASS_DISTRIBUTION, ConfusionMatrix, accuracy
from . import dense_transforms
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = FCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    Hint: Use ConfusionMatrix, ConfusionMatrix.add(logit.argmax(1), label), ConfusionMatrix.iou to compute
          the overall IoU, where label are the batch labels, and logit are the logits of your classifier.
    Hint: If you found a good data augmentation parameters for the CNN, use them here too. Use dense_transforms
    Hint: Use the log function below to debug and visualize your model
    """
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    #CHANGE TO ADAM!!

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    loss = ClassificationLoss()  #<--------------CHANGE!!!!!!!!!!!!!??
    train_data = load_dense_data('dense_data/train')
    valid_data = load_dense_data('dense_data/valid')

    for epoch in range(args.num_epoch):
    
        model.train()                                  
        loss_vals, acc_vals, vacc_vals = [], [], []     
        
        #  BEGIN TRAINING-----------------------------------------------------LOOP BEGIN
     
        for img, label in train_data:                   
        
            label = label.type(torch.LongTensor)
            #print (f'Image Dimensions:  (IN TRAIN)  is {img.shape}')
            img, label = img.to(device), label.to(device)    
            logit = model(img)                              
          
            loss_val = loss(logit, label)                  
            acc_val = accuracy(logit, label)         #get accuracy on same (128,6) logits of (128,3*64*64) batch

            loss_vals.append(loss_val.detach().cpu().numpy())   #add batch loss_val to loss_vals list (with an s)
            acc_vals.append(acc_val.detach().cpu().numpy())     #add accuracy acc_val to acc_vals list (with an s)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
                                                      
      #END TRAINING LOOP------------------------------------------------------------------------LOOP END

        avg_loss = sum(loss_vals) / len(loss_vals)
        avg_acc = sum(acc_vals) / len(acc_vals)

        model.eval()   
        
        for img, label in valid_data:                     #iterate through validation data
        
            img, label = img.to(device), label.to(device)
            vacc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(vacc_vals) / len(vacc_vals)

        print('epoch %-3d \t loss = %0.3f \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_loss, avg_acc, avg_vacc))
    
    save_model(model)


def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                             convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                  label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                  convert('RGB')), global_step, dataformats='HWC')

if __name__ == '__main__':
   
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--log_dir')
  # Put custom arguments here
  parser.add_argument('-n', '--num_epoch', type=int, default=50)
  parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
  parser.add_argument('-c', '--continue_training', action='store_true')
  args = parser.parse_args()
  train(args)
