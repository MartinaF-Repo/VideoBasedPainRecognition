import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        # Sixth convolutional layer
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        # Average adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        # x.shape = (8, 3, 112, 112)

        # First convolutional block
        #x = F.relu(self.bn1(self.conv1(x))) # x = (8, 32, 112, 112) 
        #x = F.max_pool2d(x, 2) # x = (8, 32, 56, 56)
        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x))) # x = (8, 64, 112, 112)
        x = F.max_pool2d(x, 2) # x = (8, 64, 56, 56)
        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x))) # x = (8, 128, 56, 56)
        x = F.max_pool2d(x, 2) # x = (8, 128, 28, 28)
        # Fourth convolutional block
        x = F.relu(self.bn4(self.conv4(x))) # x = (8, 256, 28, 28)
        x = F.max_pool2d(x, 2) # x = (8, 256, 14, 14)
        # Fifth convolutional block
        #x = F.relu(self.bn5(self.conv5(x))) # x = (8, 512, 14, 14)
        # Sixth convolutional block
        #x = F.relu(self.bn6(self.conv6(x))) # x = (8, 1024, 14, 14)
        # Average adaptive pooling
        x = self.adaptive_pool(x) # x = (8, 256, 2, 2)
        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # x = (8, 1024)

        return x  


class HSC(nn.Module):
    def __init__(self, num_classes, sift_vectors_length, hog_vectors_length):
        super(HSC, self).__init__()

        # Fully connected layer to transform HOG feature vectors into feature vectors of length 512 
        self.hog_linear = nn.Linear(hog_vectors_length, 512) # hog_vectors_length = 6084
        # Fully connected layer to transform SIFT feature vectors into feature vectors of length 512 
        self.sift_linear = nn.Linear(sift_vectors_length, 512) # sift_vectors_length = 1024
        # CNN
        self.cnn = CNN() # cnn_vectors_length = 1024
        # First fully connected layer for classification
        self.fc1 = nn.Linear(2048, 512)
        # Second fully connected layer for classification
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, hog_features, sift_features, images):
        # HOG feature vectors
        h = F.relu(self.hog_linear(hog_features))
        # SIFT feature vectors
        s = F.relu(self.sift_linear(sift_features))
        # CNN feature vectors
        c = self.cnn(images)

        # Feature vectors concatenation
        features = torch.cat([h, s, c], dim=1)
        
        # Classification
        x = F.relu(self.fc1(features))
        x = self.fc2(x)
        x = F.softmax(x, dim = 1)

        return x, features, None  
    



