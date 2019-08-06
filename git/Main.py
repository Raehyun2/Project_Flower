import argparse
import sklearn.model_selection
import numpy as np

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.utils.data

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from torchsummary import summary
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(np.array(y_true),np.array(y_pred))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if normalize:
        plt.savefig('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Evaluation\\confusion_normalized.jpg')
    else:
        plt.savefig('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Evaluation\\confusion.jpg')
    return ax


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            # 3 x 200 x 200
            nn.Conv2d(3, 32, 3, stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 32 x 198 x 198
            nn.Conv2d(32, 64, 3, stride = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 64 x 196 x 196
            nn.MaxPool2d(2, 2),

            # 64 x 98 x 98
            nn.Conv2d(64, 128, 3, stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 128 x 96 x 96
            nn.Conv2d(128, 256, 3, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 256 x 94 x  94
            nn.MaxPool2d(2, 2),

            # 256 x 47 x 47
            nn.Conv2d(256,512,3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        # 512 x 45 x 45

        self.avg_pool = nn.AvgPool2d(45)

        # self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Linear(512,5)

    def forward(self, x):
        feat = self.layer(x)
        flat = self.avg_pool(feat).view(feat.size(0),-1)
        # flat = self.dropout(flat)
        out = self.classifier(flat)
        return out, feat


def train(args, model, train_loader, optimizer, epoch, loss_func, min_loss):
    model.train()
    train_loss = 0
    correct = 0
    for batch_index,[image,label] in enumerate(train_loader):
        image = image.cuda()
        label = label.cuda()

        print(train_loader.dataset[0])

        out, feat = model(image)
        loss = loss_func(out,label)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()

        if batch_index % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(image), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)

    if train_loss<min_loss:
        print("Renew Model")
        min_loss = train_loss
        torch.save(model.state_dict(), 'D:\\Analysis\\Project\\Image\\Classification\\Flower\\Code\\Classification\\Model\\CNN_aug2')

    return train_loss, acc, min_loss

def test(args, model, test_loader, loss_func):
    model.eval()
    test_loss = 0
    correct = 0
    pred_save = []

    for image, label in test_loader:
        image = image.cuda()
        label = label.cuda()
        out, feat = model(image)
        loss = loss_func(out,label)
        test_loss += loss.item()
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item()
        pred_save.append(pred)

    test_loss /= len(test_loader)
    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    val_acc = 100.*correct/len(test_loader.dataset)

    return test_loss, val_acc, pred_save



def main():
    parser = argparse.ArgumentParser(description='Image')
    parser.add_argument('--Resize',type=int,default=(200,200))

    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.1)

    parser.add_argument('--log-interval', type=float, default=10)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.Resize(args.Resize),
                                    transforms.ToTensor()
                                    ])

    flower = dset.ImageFolder(root='D:\\Analysis\\Project\\Image\\Classification\\Flower\\Data', transform=transform)

    train_data,test_data,train_label,test_label = sklearn.model_selection.train_test_split(flower,
                                                                                           flower.targets,
                                                                                           stratify=flower.targets,
                                                                                           test_size = 0.25,
                                                                                           random_state=42)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=args.batch_size,num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=args.batch_size,num_workers=0)
    model = CNN().cuda()

    # model = models.resnet18(pretrained=True)
    # model = models.alexnet(pretrained=True)
    # model = models.vgg16(pretrained=True)
    # model = models.googlenet(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # num_ft = model.fc.in_features
    # model.fc = nn.Linear(num_ft,5)
    # for param in model.parameters():
    #     param.requires_grad = False
    #
    # num_ft = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ft,5)
    # for param in model.parameters():
    #     param.requires_grad = False
    # num_ft = model.classifier[6].in_features
    # model.classifier[6] = nn.Linear(num_ft,5)
    #
    # for param in model.parameters():
    #     param.requires_grad = False
    # num_ft = model.fc.in_features
    # model.fc = nn.Linear(num_ft,5)


    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    summary(model, (3,200,200))

    train_loss_plot = []
    test_loss_plot = []
    train_acc_plot = []
    test_acc_plot = []
    min_loss = 999

    for epoch in range(1, args.epochs + 1):
        [train_loss, acc, min_loss] = train(args, model, train_loader, optimizer, epoch, loss_func,min_loss)
        [test_loss, val_acc, pred_save] = test(args, model, test_loader, loss_func)

        train_loss_plot = np.append(train_loss_plot, train_loss)
        test_loss_plot = np.append(test_loss_plot, test_loss)
        train_acc_plot = np.append(train_acc_plot, acc)
        test_acc_plot = np.append(test_acc_plot, val_acc)

    if epoch < args.epochs:
        pred_save = []
    else:
        pred_save = torch.cat(tuple(pred_save)).cpu().numpy().T[0]

    plt.figure(1)
    plt.plot(train_loss_plot)
    plt.plot(test_loss_plot)
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.ylim([0, 5])
    plt.legend(['train', 'test'])
    plt.savefig('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Evaluation\\Loss.jpg')

    plt.figure(2)
    plt.plot(train_acc_plot)
    plt.plot(test_acc_plot)
    plt.title('Model ACC')
    plt.ylabel('ACC')
    plt.xlabel('epochs')
    plt.ylim([0, 100])
    plt.legend(['train', 'test'])
    plt.savefig('D:\\Analysis\\Project\\Image\\Classification\\Flower\\Result\\Evaluation\\ACC.jpg')

    test_label = torch.tensor(test_label).numpy()
    classes = np.array(flower.classes)
    plot_confusion_matrix(test_label,pred_save,classes, normalize=True)

if __name__ == '__main__':
    main()