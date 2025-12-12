import pandas as pd
from torchvision import transforms, datasets
import pandas as pd
from geometric_class import GeometricModel
from sklearn.model_selection import GridSearchCV
import os

# Function that transforms the video dataset into a sklearn DataFrame
def dataset_to_dataframe(dataset):
    cnt = 0
    frames = []
    videos = []
    labels = []
    
    for image, label in dataset:
        if cnt == 8:
            cnt = 0
            frames = []
        if cnt == 7:
            labels.append(label)
            videos.append(frames)
        frames.append(image)
        cnt += 1
        
    print(len(videos))
    print(len(videos[0]))
    print(len(labels))

    df = pd.DataFrame(videos, columns=["frame 1", "frame 2", "frame 3", "frame 4", "frame 5", "frame 6", "frame 7", "frame 8" ])
    df['target'] = labels

    return df


if __name__ == '__main__':
    # Load train and test datasets and convert them into sklearn DataFrame
    base_path = r"C:\CV Project\frames"
    train_dataset = datasets.ImageFolder(f'{base_path}/FrameTrain', transform=transforms.Compose([transforms.PILToTensor(), transforms.Resize((112, 112), antialias=True)]))
    train_dataframe = dataset_to_dataframe(train_dataset)
    test_dataset = datasets.ImageFolder(f'{base_path}/FrameTest', transform=transforms.Compose([transforms.PILToTensor(), transforms.Resize((112, 112), antialias=True)]))
    test_dataframe = dataset_to_dataframe(test_dataset)

    #dataframe.to_pickle("./dataframe.pkl")
    #dataframe = pd.read_pickle("./dataframe.pkl")
    #dataframe.to_csv("dataset.csv", sep = ',', index = False)
    #dataframe = pd.read_csv("dataset.csv")

    """
    G_model = GeometricModel(pretrained = True, epochs=20, kmeans_clusters=1024)
    X_train = train_dataframe.drop("target", axis = 1)
    y_train = train_dataframe["target"]

    G_model.fit(X_train, y_train)
    print()

    X_test = test_dataframe.drop("target", axis = 1)
    y_test = test_dataframe["target"]

    #print(G_model.predict(X_test))

    print(G_model.score(X_test, y_test))
    """
    
    # Prepare train features and labels dataframes
    X_train = train_dataframe.drop("target", axis = 1)
    y_train = train_dataframe["target"]

    # Apply GridSearch 
    parameters = {
        'epochs': [10, 15],
        'kmeans_clusters': [512, 1024, 2048],
        'lr': [0.0001, 0.001], 
        'max_keypoints_per_frame' : [300, 500, 800],
        'sift_edge_threshold' : [10, 15]
    }
    gs = GridSearchCV(estimator=GeometricModel(), param_grid=parameters, verbose=4)
    gs.fit(X_train, y_train)
    print("Best parameters: ", gs.best_params_) 
    print()
    print("Best accuracy: ", gs.best_score_)
    print()
    print("All results: ", gs.cv_results_)

    # Save results
    results = pd.DataFrame(gs.cv_results_)
    gridsearch_dir = './grid_search'
    if not os.path.exists(gridsearch_dir):
        os.makedirs(gridsearch_dir)
    results.to_csv(gridsearch_dir + "/gs_results.csv", sep = ',', index = False)
    results.to_pickle(gridsearch_dir + "/gs_results.pkl")

