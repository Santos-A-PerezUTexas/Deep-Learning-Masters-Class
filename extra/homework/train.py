import torch
import torch.nn as nn
from .models import TCN, save_model
from .utils import SpeechDataset, one_hot, load_data


def train(args):
    from os import path
    import torch.utils.tensorboard as tb
    model = TCN()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your code from prior assignments
    Hint: SGD might need a fairly high learning rate to work well here

    """

    train_data = load_data('data/train.txt',  transform=one_hot)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    Crossloss = torch.nn.CrossEntropyLoss()
    MSEloss = torch.nn.MSELoss(reduction='none')
    loss_val = 0
    
    for epoch in range(args.num_epoch):

      model.train()
      
      print (f'Epoch {epoch}, loss is {loss_val}')

      i= 0

      for batch in train_data:

        #batch is torch.Size([32, 28, 250])

        batch_data = batch[:,:,:-1]
        #batch_labels = batch[:,:,1:].argmax(dim=1)
        batch_labels = batch.argmax(dim=1)
        
        
        #batch data is one hot encoded, so this gives you the index where the one's are, e.g
        #the actual letter, but this label does not include the first column! so network
        #must learn to recreate the last column INDICES
        #so that batch_label 32,249 contains the indices corresponding to the letter


        prediction = model(batch_data)  #[:, 0, :] 
       
        print (f'1   The prediction shape  is {prediction.shape}, epoch {epoch}, batch {i}')
        print (f'2   The batch  data shape  is {batch_data.shape}, epoch {epoch}, batch {i}')
        #print (f'The batch  label  is {batch_labels.shape}, epoch {epoch}, batch {i}')
        
        i+=1  
        
        #print (f'The prediction type  is {prediction.dtype}')
        
        print (f'3    The batch_labels shape  is {batch_labels.shape}')
        

        loss_val = Crossloss(prediction, batch_labels)
        #loss_val = MSEloss(prediction, batch_labels).mean()
        
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


    save_model(model)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=120)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor(), ToHeatmap(2)])')
    parser.add_argument('-w', '--size-weight', type=float, default=0.01)

    args = parser.parse_args()
    train(args)
