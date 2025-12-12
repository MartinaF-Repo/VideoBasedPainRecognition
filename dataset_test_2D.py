import os
import sys
import argparse
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.utils.data as data
from networks.DDAM import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default=r"F:\Dataset", help='Dataset dataset path.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of class.')
    parser.add_argument('--model_path', default=r'C:\Users\dalla\Desktop\DDAMFN\checkpoints\dataset_epoch6_acc0.6842.pth')
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

class2_names = ['FrameReal', 'FrameFake']

def run_test():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()        

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
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

    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
  
    for imgs, targets in val_loader:
        #imgs = imgs.to(device)
        #targets = targets.to(device)
        #out, feat, heads = model(imgs)
        #imgs = imgs.to(device)
        imgs = imgs.unsqueeze(0) # 1,8,3,112,112
        print(imgs.shape)
        print(imgs.size)
        imgs= imgs.permute(0, 2, 1, 3, 4).to(device) #1,3,8,112,112
        print(f"--- imgs shape: {imgs.size}")
        print(imgs.size)
        targets = targets.to(device)
                        
        out, feat, heads = model(imgs)    
        _, predicts = torch.max(out, 1)
        correct_num  = torch.eq(predicts, targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out.size(0)
        
        if iter_cnt == 0:
            all_predicted = predicts
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicts), 0)
            all_targets = torch.cat((all_targets, targets), 0)                  
        iter_cnt += 1        

    acc = bingo_cnt.float() / float(sample_cnt)
    acc = np.around(acc.numpy(), 4)

    print("Validation accuracy: %.4f" % acc)
    
    # Compute confusion matrix
    matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    # Plot normalized confusion matrix
    plot_confusion_matrix(matrix, classes=class2_names, normalize=True, title='Confusion Matrix (acc: %0.2f%%)' % (acc*100))

    plt.savefig(os.path.join('checkpoints', "dataset"+"_acc"+str(acc)+".png"))
    plt.close()            

if __name__ == "__main__":                
    run_test()
