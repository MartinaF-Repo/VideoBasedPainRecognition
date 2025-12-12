import os
import shutil
import random

def split_dataset(source_folder, train_folder, test_folder, val_folder, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    # Creare le cartelle di destinazione se non esistono
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # Listare tutti i file nella cartella di origine
    files = os.listdir(source_folder)
    
    # Mischiare i file
    random.shuffle(files)
    
    # Calcolare i punti di divisione
    total_files = len(files)
    train_split = int(total_files * train_ratio)
    test_split = int(total_files * (train_ratio + test_ratio))
    
    # Dividere i file in train, test e val
    train_files = files[:train_split]
    test_files = files[train_split:test_split]
    val_files = files[test_split:]
    
    # Spostare i file nelle rispettive cartelle
    for file in train_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(train_folder, file))
    
    for file in test_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(test_folder, file))
    
    for file in val_files:
        shutil.move(os.path.join(source_folder, file), os.path.join(val_folder, file))

# Definire i percorsi delle cartelle
base_source_folder = 'C:\\Users\\Nico\\Desktop\\Dataset'
train_folder = os.path.join(base_source_folder, 'Train')
test_folder = os.path.join(base_source_folder, 'Test')
val_folder = os.path.join(base_source_folder, 'Val')

# Eseguire la divisione per ciascuna categoria
split_dataset(os.path.join(base_source_folder, 'VideoReal'), os.path.join(train_folder, 'VideoReal'), os.path.join(test_folder, 'VideoReal'), os.path.join(val_folder, 'VideoReal'))
split_dataset(os.path.join(base_source_folder, 'VideoFake'), os.path.join(train_folder, 'VideoFake'), os.path.join(test_folder, 'VideoFake'), os.path.join(val_folder, 'VideoFake'))
