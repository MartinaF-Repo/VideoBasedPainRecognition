import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from networks_3D.DDAM3D import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default=r"F:\Dataset", help='Dataset dataset path.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of class.')
    parser.add_argument('--model_path', default=r'C:\Users\dalla\Desktop\DDAMFN\checkpoints_DDAM3D\dataset_epoch33_acc0.7719.pth')
    return parser.parse_args()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt)+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()

class2_names = ['FrameFake', 'FrameReal']

def video_loader(loader):
        labels = []
        frames = []
        num_element = 0

        for frame, label in loader:
            if num_element >= len(labels):
                labels.append(label)
                frames.append([])

            frames[num_element].append(frame)

            if len(frames[num_element]) == 8:
                num_element += 1

        new_trainloader = list(zip(frames, labels))
        return new_trainloader

def run_test():
    args = parse_args()
    device = torch.device("cuda:0")

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict']) #strict=False
    model.to(device)
    model.eval()        

    data_transforms_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])     
                                                                      
    val_dataset = datasets.ImageFolder(f'{args.aff_path}/FrameVal', transform=data_transforms_val)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,  
                                             pin_memory=True)

    new_valloader = video_loader(val_loader)

    correct_sum = 0
    all_targets = []
    all_predicted = []
  
    for videos, targets in new_valloader:
        
        videos = torch.stack(videos, dim=0) #[8,1,3,128,128]

        videos= videos.permute(1,2,0,3,4).to(device) #[1,3,8,128,128]
        targets = targets.to(device)
                        
        out, feat, heads = model(videos)    
        _, predicts = torch.max(out, 1)
        
        if(predicts[0]==targets[0]):
            correct_sum += 1
        
        all_predicted.append(int(predicts[0]))
        all_targets.append(int(targets[0]))        

    acc = float(correct_sum) / float(len(new_valloader))
    acc = np.around(np.array(acc), 4)

    print("Validation accuracy: %.4f" % acc)
    
    # Compute confusion matrix
    all_targets = torch.tensor(all_targets)
    all_predicted = torch.tensor(all_predicted)
    matrix = confusion_matrix(all_targets.cpu().numpy(), all_predicted.cpu().numpy())
    print(matrix)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    # Plot normalized confusion matrix
    plot_confusion_matrix(matrix, classes=class2_names, normalize=True,
                          title='Confusion Matrix (acc: %0.2f%%)' % (acc * 100))

    plt.savefig(os.path.join('checkpoints', "dataset" + "_acc" + str(acc) + ".png"))
    plt.close()        

if __name__ == "__main__":                
    run_test()
