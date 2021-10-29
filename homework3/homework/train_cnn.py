from .models import CNNClassifier, save_model
from .utils import ConfusionMatrix, load_data, LABEL_NAMES
import torch
import torchvision
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)

    """
    Your code here, modify your HW1 / HW2 code
    
    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNClassifier().to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th')))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    import inspect
    transform = eval(args.transform,
                     {k: v for k, v in inspect.getmembers(torchvision.transforms) if inspect.isclass(v)})
    train_data = load_data('data/train', transform=transform, num_workers=4)
    valid_data = load_data('data/valid', num_workers=4)

    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        confusion = ConfusionMatrix(len(LABEL_NAMES))
        for img, label in train_data:
            if train_logger is not None:
                train_logger.add_images('augmented_image', img[:4])
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label)
            confusion.add(logit.argmax(1), label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1

        if train_logger:
            train_logger.add_scalar('accuracy', confusion.global_accuracy, global_step)
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.imshow(confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(confusion.per_class.size(0)):
                for j in range(confusion.per_class.size(1)):
                    ax.text(j, i, format(confusion.per_class[i, j], '.2f'),
                            ha="center", va="center", color="black")
            train_logger.add_figure('confusion', f, global_step)

        model.eval()
        val_confusion = ConfusionMatrix(len(LABEL_NAMES))
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            val_confusion.add(model(img).argmax(1), label)

        if valid_logger:
            valid_logger.add_scalar('accuracy', val_confusion.global_accuracy, global_step)
            import matplotlib.pyplot as plt
            f, ax = plt.subplots()
            ax.imshow(val_confusion.per_class, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(val_confusion.per_class.size(0)):
                for j in range(val_confusion.per_class.size(1)):
                    ax.text(j, i, format(val_confusion.per_class[i, j], '.2f'),
                            ha="center", va="center", color="black")
            valid_logger.add_figure('confusion', f, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, confusion.global_accuracy,
                                                                    val_confusion.global_accuracy))
        save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=75)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-t', '--transform',
                        default='Compose([ColorJitter(0.9, 0.9, 0.9, 0.1), RandomHorizontalFlip(), ToTensor()])')

    args = parser.parse_args()
    train(args)


