from torchvision import transforms, datasets
import torch
import torch.utils.data as data
import os
import sys
import pickle
import random
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from torch.backends import cudnn
from networks_geometric import GFE
from networks_geometric import HSC

retrieval_list = []
eps = sys.float_info.epsilon
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default=r"C:\CV Project\frames", help='Dataset path.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=15, help='Total training epochs.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes.')
    return parser.parse_args()

def run_training():
    random.seed(42)
    args = parse_args()

    device = torch.device("cuda:0")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    # Create the models
    GFE_model = GFE.GeometricFeatureExtraction(pretrained=False, n_clusters=1024, max_keypoints_per_frame=800, sift_edge_threshold=10)
    HSC_model = HSC.HSC(num_classes=args.num_class, sift_vectors_length=1024, hog_vectors_length=6084)
    HSC_model.to(device)

    # Load train and validation dataset and divide them into batches
    train_dataset = datasets.ImageFolder(f'{args.aff_path}/FrameTrain', transform=transforms.Compose([transforms.PILToTensor(), transforms.Resize((112, 112), antialias=True)]))
    val_dataset = datasets.ImageFolder(f'{args.aff_path}/FrameVal', transform=transforms.Compose([transforms.PILToTensor(), transforms.Resize((112, 112), antialias=True)]))

    print('Whole train set size:', len(train_dataset))
    print('Validation set size:', len(val_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=False,
                                               pin_memory=True)


    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    # Fit the features extraction model on the training data
    GFE_model.BoK.fit(train_loader)

    # Extract HOG and SIFT features from both the datasets and shuffle the batches
    new_train_loader = GFE_model.get_features(train_loader)
    random.shuffle(new_train_loader)
    new_val_loader = GFE_model.get_features(val_loader)
    random.shuffle(new_val_loader)

    # Define loss and optimizer
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(HSC_model.parameters(), args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

    best_acc = 0

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    # Training loop
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        HSC_model.train()

        for (hog_imgs, sift_imgs, cnn_imgs, targets) in new_train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            hog_imgs = hog_imgs.to(device)
            sift_imgs = torch.FloatTensor(sift_imgs).to(device)
            cnn_imgs = cnn_imgs.to(device)
            targets = targets.to(device)

            # Compute batch feature vectors and class scores
            out, feat, heads = HSC_model(hog_imgs, sift_imgs, cnn_imgs.float())
            
            loss = criterion_cls(out, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum().cpu()
            correct_sum += correct_num

            # If the video has been classified correctly, save the mean of the feature vectors of its frames for the retrieval algorithm
            if (epoch == args.epochs):
                if (predicts.sum() == targets.sum()):
                    feat_mean = torch.mean(feat, dim=0)
                    retrieval_list.append((feat_mean, predicts[0]))

        # Compute training accuracy
        acc = correct_sum.float() / len(train_dataset)
        running_loss /= iter_cnt
        train_accs.append(np.around(acc.numpy(), 4))
        train_losses.append(np.around(np.array(running_loss), 4))
        tqdm.write(
            f'[Epoch {epoch}] Training accuracy: {acc:.4f}. Loss: {running_loss:.3f}. LR {optimizer.param_groups[0]["lr"]:.6f}')

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            HSC_model.eval()
            for (hog_imgs, sift_imgs, cnn_imgs, targets) in new_val_loader:
                hog_imgs = hog_imgs.to(device)
                sift_imgs = torch.FloatTensor(sift_imgs).to(device)
                cnn_imgs = cnn_imgs.to(device)
                targets = targets.to(device)

                # Compute batch feature vectors and class scores
                out, feat, heads = HSC_model(hog_imgs, sift_imgs, cnn_imgs.float())

                loss = criterion_cls(out, targets)

                running_loss += loss.item()
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

            running_loss /= iter_cnt
            scheduler.step()

            # Compute validation accuracy
            acc = bingo_cnt.float() / sample_cnt
            acc = np.around(acc.numpy(), 4)
            val_accs.append(np.around(np.array(acc), 4))
            val_losses.append(np.around(np.array(running_loss), 4))
            best_acc = max(acc, best_acc)
            tqdm.write(f"[Epoch {epoch}] Validation accuracy: {acc:.4f}. Loss: {running_loss:.3f}")
            tqdm.write(f"Best accuracy: {best_acc}")

            # Save model
            checkpoint_dir = './geometric_checkpoints'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            
            torch.save({'iter': epoch,
                            'model_state_dict': HSC_model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoint_dir, f"frame_classification_epoch{epoch}_acc{acc:.4f}.pth"))
            tqdm.write('Model saved.')

    # Save video feature vectors for retrieval algorithm
    with open('./pretrained/retrieval_features.pkl', 'wb') as f:
                    pickle.dump(retrieval_list, f)

    print("Train losses: ", train_losses)
    print("Train accs: ", train_accs)
    print("Val losses: ", val_losses)
    print("Val accs: ", val_accs)

if __name__ == "__main__":
    run_training()
