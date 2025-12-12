from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms, datasets
import os

dataset_path = r"F:\DatasetOLD"
class_one = ['FrameTrain']


# Funzione per sharpening con OpenCV
def sharpen_image_opencv(img):
    img = np.array(img)
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3,-0.5],
                       [0, -0.5, 0]])
    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    return Image.fromarray(img)

# Creazione della trasformazione di sharpening
sharpen_transform_opencv = transforms.Lambda(lambda img: sharpen_image_opencv(img))

# Funzione per creare una nuova trasformazione con parametri fissi
def create_transform():
    # Genera parametri fissi per la trasformazione
    rotation_angle = np.random.uniform(-15, 15)  # Angolo fisso
    translation_x = np.random.uniform(-0.1, 0.1)  # Traslazione fissa in x
    translation_y = np.random.uniform(-0.1, 0.1)  # Traslazione fissa in y
    scale_factor = np.random.uniform(0.9, 1.1)  # Scalatura fissa

    # Funzione per applicare la trasformazione affine fissa
    def apply_affine_transform(img):
        return transforms.functional.affine(img, angle=rotation_angle, translate=(translation_x, translation_y), scale=scale_factor, shear=0)

    return transforms.Compose([
        sharpen_transform_opencv,
        transforms.Lambda(apply_affine_transform),
        transforms.ToTensor(),
    ])

for subdir in class_one:
    path = os.path.join(dataset_path, subdir)  # Correzione del percorso

    train_dataset = datasets.ImageFolder(path)
    cnt = 8
    imgs_list = list(train_dataset.imgs)
    img_index = 0  # Inizializza l'indice per iterare su imgs_list

    while img_index < len(imgs_list):
        # Crea una nuova trasformazione con parametri fissi per ogni nuovo video (ogni gruppo di 8 frame)
        data_transforms = create_transform()

        # Applica la stessa trasformazione agli 8 frame successivi
        for _ in range(8):  # Processa 8 frame
            if img_index >= len(imgs_list):
                break  # Esci se non ci sono più immagini
            
            m, t = imgs_list[img_index]
            
            # Carica l'immagine senza trasformazioni
            img, target = train_dataset[img_index]

            # Applica le trasformazioni
            img_transformed = data_transforms(img)

            # Salva l'immagine trasformata
            img_save_path = f"{m[:-5]}{cnt}_M.png"
            img = transforms.ToPILImage()(img_transformed)
            img.save(img_save_path)

            cnt += 1
            img_index += 1  # Avanza al prossimo frame

        # Dopo aver elaborato 8 frame, passa al prossimo video
        cnt = 8  # Reset cnt per il prossimo gruppo di 8 frame
