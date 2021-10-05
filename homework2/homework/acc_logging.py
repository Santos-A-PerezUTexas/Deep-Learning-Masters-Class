#Homework 2
#Oct 4, 2021

from os import path
import torch
import torch.utils.tensorboard as tb
import tempfile
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math


#log_dir = tempfile.mkdtemp()


def test_logging(train_logger, valid_logger):

    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epoch
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """

    #train_logger.add_scalar (below) : It will plot just one graph
    #train_logger.add_scalars (below): It will plot multi graphs at once
    

    # This is a strongly simplified training loop
    i=0
    x= torch.zeros(20,10)
    t=0
    y= torch.zeros(10,10)

    for epoch in range(10):
    
        torch.manual_seed(epoch)
           
        for iteration in range(20):
            
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            
            x[iteration] = dummy_train_accuracy

            train_logger.add_scalar('accuracy',x.mean(), global_step=i)
            train_logger.add_scalar('loss', dummy_train_loss, global_step=i)
            i+=1 
      
        torch.manual_seed(epoch)
        
        for iteration in range(10):
            
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            y[iteration] = dummy_validation_accuracy
            
            valid_logger.add_scalar('accuracy', y.mean(), global_step=t)
            t+=1 
        print (epoch, y.mean())
    
    



    
if __name__ == "__main__":
    from argparse import ArgumentParser

    print ("PROGRAM RUNNING")
    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
