import os
import sys
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from networks.DDAM import DDAMNet
import torch.nn.functional as F

eps = sys.float_info.epsilon
torch.cuda.empty_cache()

def parse_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default=r"F:\Dataset", help='Dataset path.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=20, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention heads.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes.')
    return parser.parse_args() 

class ImbalancedDatasetSampler(data.sampler.Sampler):
    def __init__(self, dataset, indices: list = None, num_samples: int = None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset)
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if isinstance(dataset, datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torch.utils.data.Subset):
            return [dataset.dataset.imgs[i][1] for i in dataset.indices]
        else:
            raise NotImplementedError

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class AttentionLoss(nn.Module):
    def __init__(self):
        super(AttentionLoss, self).__init__()
    
    def forward(self, x):
        num_head = len(x)
        loss = 0
        cnt = 0
        if num_head > 1:
            for i in range(num_head-1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    cnt += 1
                    loss += mse
            loss = cnt / (loss + eps)
        else:
            loss = 0
        return loss

def run_training():
    args = parse_args()

    device = torch.device("cuda:0")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
            ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])

    train_dataset = datasets.ImageFolder(f'{args.aff_path}/FrameTrain', transform=data_transforms)   

    print('Whole train set size:', len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               #sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle=False, 
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])      

    val_dataset = datasets.ImageFolder(f'{args.aff_path}/FrameVal', transform=data_transforms_val)  

    print('Validation set size:', len(val_dataset))
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,  
                                             pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    criterion_at = AttentionLoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)
    
    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device) # 8, 3, 128, 128
            targets = targets.to(device) # torch.Size([8])
                        
            out, feat, heads = model(imgs) #[8,2]
            

            loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            
            if(predicts.sum() > 4):
                pred=1
            else:
                pred=0
            
            if(pred==targets[0]):
                correct_sum += 1
        
            
        acc = float(correct_sum) / (len(train_dataset)/8)
        running_loss /= iter_cnt
        #da cambiare anche nel validation
        tqdm.write(f'[Epoch {epoch}] Training accuracy: {acc:.4f}. Loss: {running_loss:.3f}. LR {optimizer.param_groups[0]["lr"]:.6f}')
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                out, feat, heads = model(imgs)

                loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)

                running_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
            running_loss /= iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float() / sample_cnt
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)
            tqdm.write(f"[Epoch {epoch}] Validation accuracy: {acc:.4f}. Loss: {running_loss:.3f}")
            tqdm.write(f"Best accuracy: {best_acc}")

            # Assicurati che la directory checkpoints esista
            checkpoint_dir = './checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            if args.num_class == 2 and acc > 0.665:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                            os.path.join(checkpoint_dir, f"dataset_epoch{epoch}_acc{acc:.4f}.pth"))
                tqdm.write('Model saved.')
        
if __name__ == "__main__":                    
    run_training()