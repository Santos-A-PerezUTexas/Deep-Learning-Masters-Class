#HOMEWORK 3
#HOMEWORK 3
#https://bit.ly/3AaUm3l 


from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb
import torchvision.transforms as T

def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
        
    rgb_mean = (0.4914, 0.4822, 0.4465)
    rgb_std = (0.2023, 0.1994, 0.2010)

    transform_train = T.transforms.Compose([
      T.transforms.RandomCrop(32, padding=4),
      T.transforms.RandomHorizontalFlip(),
      T.transforms.ToTensor(),
      T.transforms.Normalize(rgb_mean, rgb_std),
    ])

    """
    Your code here, modify your HW1 / HW2 code
    ADD LOGGING CODE
    
    """
    model = CNNClassifier()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = ClassificationLoss()

    #ADD TRANSFORMS HERE?
    train_data = load_data('data/train', transform=transform_train)
    valid_data = load_data('data/valid')

    for epoch in range(args.num_epoch):
    
        model.train()                           
        
        loss_vals, acc_vals, vacc_vals = [], [], []     
        
        #  BEGIN TRAINING-----------------------------------------------------LOOP BEGIN
     
        for img, label in train_data:                   
        
        
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











if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
