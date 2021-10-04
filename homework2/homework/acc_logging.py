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


    for step in range(-360, 360):
      angle_rad = step * math.pi / 180
      train_logger.add_scalar('sin', math.sin(angle_rad), step)
      train_logger.add_scalar('cos', math.cos(angle_rad), step)
      train_logger.add_scalars('sin and cos', {'sin': math.sin(angle_rad), 'cos': math.cos(angle_rad)}, step)
      train_logger.close()



    # This is a strongly simplified training loop
    for epoch in range(10):
    
        torch.manual_seed(epoch)
        
        
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            print(iteration)
            #logger = tb.SummaryWriter(log_dir, flush_secs=1)
            train_logger.add_scalar('first/ACCURACY', dummy_train_accuracy[epoch], global_step=iteration)
            #raise NotImplementedError('Log the training loss')
        train_logger.add_scalar('first/error', dummy_train_loss, global_step=epoch)


        #test
        #raise NotImplementedError('Log the training accuracy')
        torch.manual_seed(epoch)



"""
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)

        raise NotImplementedError('Log the validation accuracy')
"""

if __name__ == "__main__":
    from argparse import ArgumentParser

    print ("PROGRAM RUNNING")
    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
