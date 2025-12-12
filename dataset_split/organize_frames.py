import os
import shutil
from collections import defaultdict

def organize_frames_by_prefix(base_dir):
    # Definire le macrocartelle
    macro_folders = ['FrameTrain', 'FrameTest', 'FrameVal']
    sub_folders = ['FrameReal', 'FrameFake']

    for macro_folder in macro_folders:
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(base_dir, macro_folder, sub_folder)
            
            if not os.path.exists(sub_folder_path):
                print(f"La cartella {sub_folder_path} non esiste. Saltando...")
                continue

            # Raggruppare i file per prefisso
            frame_groups = defaultdict(list)
            for file_name in os.listdir(sub_folder_path):
                if file_name.endswith(".jpg"):
                    # Estrarre il prefisso del file (tutto fino al primo underscore prima del contatore)
                    prefix = "_".join(file_name.split("_")[:-1])
                    frame_groups[prefix].append(file_name)
            
            # Creare una cartella per ciascun gruppo di frame e spostarli
            for prefix, files in frame_groups.items():
                video_folder = os.path.join(sub_folder_path, prefix)
                
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)

                for file_name in files:
                    src = os.path.join(sub_folder_path, file_name)
                    dst = os.path.join(video_folder, file_name)
                    shutil.move(src, dst)

# Esegui la funzione
organize_frames_by_prefix(r"F:\DatasetNew")
