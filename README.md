# PainDetection
## A Comparative Study of Architectures with Different Complexity Levels

This repository contains the implementation and evaluation of multiple architectures for automatic pain detection from visual data, with a focus on comparing models of increasing computational and representational complexity.

The project includes geometry-based methods, 3D deep learning architectures, retrieval-based approaches, and extensive experimentation utilities such as grid search, data augmentation, and evaluation artifacts.

---

## Project Structure

### Geometry-Based Model

This module implements pain detection using geometric and handcrafted facial features, combined with learning-based classification.

**Core components**
- DDAMFN/networks_geometric/GFE.py  
  Geometric Feature Extractor (GFE)
- DDAMFN/networks_geometric/HSC.py  
  Hierarchical Shape Coding (HSC)

**Training and evaluation**
- DDAMFN/geometric_class.py  
- DDAMFN/geometric_train.py  
- DDAMFN/geometric_test.py  
- DDAMFN/geometric_train_voting.py  
- DDAMFN/geometric_test_voting.py

**Pretrained assets**
- DDAMFN/pretrained/kmeans_model.pkl  
  K-means model used for geometric feature clustering

---

### 3D Deep Learning Models

This module contains spatiotemporal deep learning architectures operating on video sequences for pain recognition.

**Network architectures**
- DDAMFN/networks_3D/DDAM3D.py  
- DDAMFN/networks_3D/MFN3D.py

**Dataset handling**
- DDAMFN/dataset_train_3D.py  
- DDAMFN/dataset_test_3D.py

**Model checkpoints**
- DDAMFN/checkpoints_DDAM3D/  
  Saved weights for trained 3D models

---

### Retrieval-Based Algorithm

A feature-retrieval approach used for comparison with classification-based models.

- DDAMFN/RetrievalAlgorithm.py  
- DDAMFN/pretrained/retrieval_feature.pkl  

---

### Hyperparameter Optimization

Grid search utilities used to tune model parameters.

- DDAMFN/grid_search.py  
- DDAMFN/grid_search/gs_results.csv  
- DDAMFN/grid_search/gs_results.pkl  

---

### Data Augmentation

Image and frame-level augmentation techniques applied during training.

- DDAMFN/image_processing.py

---

### Evaluation Outputs

**Confusion matrices**
- DDAMFN/confusion_matrices/

**Dataset splits and extracted frames**
- DDAMFN/dataset_split/  
- DDAMFN/frames/

---

## Notes

- This project was developed as a comparative research study, not as a production-ready system.
- Multiple architectures are implemented to evaluate trade-offs between model complexity, interpretability, and performance.

## Authors

Developed by a team of three contributors.
