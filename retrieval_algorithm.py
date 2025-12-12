import os
import sys
import argparse
import numpy as np
import torch
import numpy
from torch import nn
import pickle
from torchvision import transforms, datasets
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from networks_geometric import GFE
from networks_geometric import HSC
from concurrent.futures import ThreadPoolExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default=r"C:\CV Project\frames",
                        help='Dataset path.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes.')
    parser.add_argument('--model_path', default = r'C:\CV Project\geometric_checkpoints\dataset_epoch10_acc0.9123.pth')
                        
    return parser.parse_args()

# Function that plots the confusion matrix
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
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


class_names = ['FrameFake', 'FrameReal']

# Function that computes the cosine similarity between two vectors
def compare(vector,feat):
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    return cos(vector, feat)

def run_test():
    args = parse_args()
    device = torch.device("cuda:0")

    # Create the models
    HSC_model = HSC.HSC(num_classes=args.num_class, sift_vectors_length=1024, hog_vectors_length=6084)
    GFE_model = GFE.GeometricFeatureExtraction(pretrained=True, n_clusters=1024, max_keypoints_per_frame=800, sift_edge_threshold=10)

    # Load pretrained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    HSC_model.load_state_dict(checkpoint['model_state_dict'])
    HSC_model.to(device)
    HSC_model.eval()

    # Load test dataset and divide it into batches
    test_dataset = datasets.ImageFolder(f'{args.aff_path}/FrameTest', transform=transforms.Compose([transforms.PILToTensor(), transforms.Resize((112, 112), antialias=True)]))

    print('Test set size:', test_dataset.__len__())

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0

    # Compute HOG and SIFT features
    new_test_loader = GFE_model.get_features(test_loader)

    # Load saved feature vectors with labels
    with open('./pretrained/retrieval_features.pkl', 'rb') as f:
        vector = pickle.load(f)
  
    all_predicted=[]
    all_targets=[]
    correct_sum=0
    for (hog_imgs, sift_imgs, cnn_imgs, targets) in new_test_loader:
        hog_imgs = hog_imgs.to(device)
        sift_imgs = torch.FloatTensor(sift_imgs).to(device)
        cnn_imgs = cnn_imgs.to(device)
        targets = targets.to(device)
        
        # Compute batch feature vectors
        out, feat, heads = HSC_model(hog_imgs, sift_imgs, cnn_imgs.float())
        # Compute the mean of the feature vectors to use it as the feature vector of the video
        feat_mean = torch.mean(feat, dim=0)
        feat_mean = feat_mean.repeat(len(vector), 1)

        # Compute cosine similarity between the current test feature vector and the saved ones
        with ThreadPoolExecutor() as executor:
            descriptors = list(
                executor.map(compare, [v[0] for v in vector], [f for f in feat_mean] ))
       
        # Extract the 10 most similar feature vectors
        labeled_scores = list(zip(descriptors, [v[1] for v in vector]))
        top_scores = sorted(labeled_scores, key=lambda labeled_scores: labeled_scores[0], reverse=True)
        top_scores = top_scores[:10]
        
        # Apply majority voting to classify the current test feature vector
        if(sum([l[1] for l in top_scores]) > 5):
            pred = 1
        else:
            pred = 0

        if(pred == targets[0]):
            correct_sum += 1   

        all_predicted.append(pred)
        all_targets.append(int(targets[0]))
        iter_cnt += 1

    # Compute test accuracy
    acc = float(correct_sum) / float(len(new_test_loader))
    acc = np.around(np.array(acc), 4)

    print("Test accuracy: %.4f" % acc)

    # Compute confusion matrix
    all_targets = torch.tensor(all_targets)
    all_predicted = torch.tensor(all_predicted)
    matrix = confusion_matrix(all_targets.cpu().numpy(), all_predicted.cpu().numpy())
    print(matrix)
    np.set_printoptions(precision=2)
    plt.figure(figsize=(10, 8))
    # Plot normalized confusion matrix
    plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                          title='Confusion Matrix (acc: %0.2f%%)' % (acc * 100))
    
    confusion_matrix_dir = './confusion_matrices'
    if not os.path.exists(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir)

    plt.savefig(os.path.join('confusion_matrices', "retrieval_algorithm_test" + "_acc" + str(acc) + ".png"))
    plt.close()


if __name__ == "__main__":
    run_test()

