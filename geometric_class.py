from networks_geometric import GFE
from networks_geometric import HSC
import sys
import torch
import random
from torchvision import transforms, datasets
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
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

eps = sys.float_info.epsilon
torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=15, help='Total training epochs.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes.')
    return parser.parse_args()

# Class that implements the geometric with voting classification model in a way which is sklearn compatible
class GeometricModel(BaseEstimator, ClassifierMixin):
    def __init__(self, pretrained = False, epochs = 15, lr = 0.001, kmeans_clusters = 2048, max_keypoints_per_frame = 500, sift_edge_threshold = 15):
        self.args = parse_args()
        self.device = torch.device("cuda:0")
        self.epochs = epochs
        self.lr = lr
        self.pretrained = pretrained
        self.kmeans_clusters = kmeans_clusters
        self.max_keypoints_per_frame = max_keypoints_per_frame
        self.sift_edge_threshold = sift_edge_threshold
        random.seed(42)
        # Create the models
        self.GFE_model = GFE.GeometricFeatureExtraction(pretrained=self.pretrained, n_clusters=self.kmeans_clusters, max_keypoints_per_frame=self.max_keypoints_per_frame, sift_edge_threshold=self.sift_edge_threshold)
        self.HSC_model = HSC.HSC(num_classes=self.args.num_class, sift_vectors_length=self.kmeans_clusters, hog_vectors_length=6084)

    def fit(self, X, y):
        
        videos = []
        for _, video in X.iterrows():
            videos.append(torch.stack([el for el in video.values]))

        labels = [torch.tensor(el).repeat(1, 8).flatten() for el in y.values]

        labeled_videos = list(zip(videos, labels))

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.enabled = True

        self.HSC_model.to(self.device)
        if self.pretrained == False:
            # Fit the features extraction model on the training data
            self.GFE_model.BoK.fit(labeled_videos)

        # Extract HOG and SIFT features and shuffle the batches
        train_dataset = self.GFE_model.get_features(labeled_videos)
        random.shuffle(train_dataset)

        # Define loss and optimizer
        criterion_cls = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.HSC_model.parameters(), self.lr, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6)

        for epoch in tqdm(range(1, self.epochs + 1)):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            self.HSC_model.train()

            for (hog_imgs, sift_imgs, cnn_imgs, targets) in train_dataset:
                iter_cnt += 1
                optimizer.zero_grad()
                hog_imgs = hog_imgs.to(self.device)
                sift_imgs = torch.FloatTensor(sift_imgs).to(self.device)
                cnn_imgs = cnn_imgs.to(self.device)
                targets = targets.to(self.device)

                # Compute batch feature vectors and class scores
                out, feat, heads = self.HSC_model(hog_imgs, sift_imgs, cnn_imgs.float())
                
                loss = criterion_cls(out, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicts = torch.max(out, 1)

                # Apply majority voting to classify the video
                if(predicts.sum() > 4):
                    pred=1
                else:
                    pred=0
                
                if(pred==targets[0]):
                    correct_sum += 1
            
            # Compute training accuracy
            acc = float(correct_sum) / len(train_dataset)
            running_loss /= iter_cnt
            tqdm.write(
                f'[Epoch {epoch}] Training accuracy: {acc:.4f}. Loss: {running_loss:.3f}. LR {optimizer.param_groups[0]["lr"]:.6f}')
            
            with torch.no_grad():
                scheduler.step()

        return self

    def predict(self, X):

        videos = []
        for _, video in X.iterrows():
            videos.append(torch.stack([el for el in video.values]))

        self.HSC_model.to(self.device)
        self.HSC_model.eval()

        # Extract HOG and SIFT features and shuffle the batches
        test_dataset = self.GFE_model.get_features(videos, labeled=False)

        iter_cnt = 0
        all_predicted = []

        for (hog_imgs, sift_imgs, cnn_imgs) in test_dataset:
            hog_imgs = hog_imgs.to(self.device)
            sift_imgs = torch.FloatTensor(sift_imgs).to(self.device)
            cnn_imgs = cnn_imgs.to(self.device)

            # Compute batch feature vectors and class scores
            out, feat, heads = self.HSC_model(hog_imgs, sift_imgs, cnn_imgs.float())

            _, predicts = torch.max(out, 1)

            # Apply majority voting to classify the video
            if(predicts.sum() > 4):
                pred=1
            else:
                pred=0

            all_predicted.append(pred)

            iter_cnt += 1

        return np.array(all_predicted)

    def score(self, X, y):
        
        predicts = self.predict(X)

        acc = ((predicts == y.values).sum()) / (y.values.shape[0])

        print("Accuracy score: ", acc)

        return acc
    
    """
    def get_params(self, deep):

        params = {
            "pretrained" : self.pretrained,
            "epochs" : self.epochs,
            "kmeans_clusters" : self.kmeans_clusters,
            "max_keypoints_per_frame" : self.max_keypoints_per_frame,
            }
        
        return params
    
    def set_params(self, params):
        
        self.pretrained = params["pretrained"]
        self.epochs = params["epochs"]
        self.kmeans_clusters = params["kmeans_clusters"]
        self.max_keypoints_per_frame = params["max_keypoints_per_frame"]

        return self

    """






    