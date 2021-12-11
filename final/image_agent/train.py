from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = model.to(device)

    print(model)

    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))

    loss = torch.nn.L1Loss()   #mean?
    #loss = torch.nn.MSELoss(reduce='mean')
    #loss= torch.nn.CrossEntropyLoss()  #render_data instance
    #loss = torch.nn.BCEWithLogitsLoss(reduction='none')  #render_data instance, tested

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    import inspect
    transform = eval(args.transform, {k: v for k, v in inspect.getmembers(dense_transforms) if inspect.isclass(v)})

    train_data = load_data(transform=transform, num_workers=args.num_workers)
   
    global_step = 0
    for epoch in range(args.num_epoch):

        model.train()
        losses = []

        print (len(train_data))

        for img, label in train_data:
            

            #print ("\n\n IN TRAIN, this is img,  label," , img.shape, label.shape)

            img, label = img.to(device), label.to(device)

            
            h, w = img.size()[2], img.size()[3]


            pred  = model(img)

            #print ("\n\n\n GOT A PREDICTION............., size", pred.shape)


            x,y = label.chunk(2, dim=1)

            #xy = torch.cat((x, y),  dim=1)  #for -1...1 coords prediction
            xy = torch.cat((x.clamp(min=0.0,max=w),y.clamp(min=0.0,max=h)),dim=1) #for 300..400

            xy = xy.to(device)

            loss_val = loss(pred, xy)
           
            #loss_val = loss(pred[:,0, :, :], label.float()).mean()   #use for render_data instance
            #print ("\n\n SAMPLE RENDER_DATA PREDICTION, LABEL:", torch.ceil(pred[0, 0, :, :]), label[0])
            #print ("\n\n SAMPLE RENDER_DATA MEAN DIFFERENCE", (torch.ceil(pred[0, 0, :, :])-label[0]).mean() )
            #print ("\n\n\n LOSS VALUE.............", loss_val)
   
            #print ("\n Sample Predicted point is .....", pred[0])
            #print ("Sample Actual point is: ", label[0])

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
                if global_step % 100 == 0:
                    log(train_logger, img, label, pred, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
            
            losses.append(loss_val.detach().cpu().numpy())
        
        avg_loss = np.mean(losses)
        if train_logger is None:
            print('epoch %-3d \t loss = %0.3f' % (epoch, avg_loss))
        save_model(model)

    save_model(model)

def log(logger, img, label, pred, global_step):
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-w', '--num_workers', type=int, default=4)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform', default='Compose([ColorJitter(0.2, 0.5, 0.5, 0.2), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)
