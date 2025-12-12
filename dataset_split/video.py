import os
import cv2
import numpy as np

def extract_frames(video_path, output_folder, num_frames=8, margin=0.2):
    """
    Estrae `num_frames` frame da un video e li salva nella cartella `output_folder`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definire i punti di inizio e fine per evitare l'inizio e la fine
    start_frame = int(total_frames * (margin + 0.20))
    end_frame = int(total_frames * (1 - margin))

    # Calcolare gli indici dei frame da estrarre
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)

    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count in frame_indices:
            frame_filename = os.path.join(output_folder, f'{os.path.splitext(os.path.basename(video_path))[0]}_frame_{count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            count += 1

        frame_count += 1

    cap.release()

def process_videos(source_folder, output_folder):
    """
    Estrae frame da tutti i video nella cartella `source_folder` e salva i frame in `output_folder`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Elenco di tutti i file nella cartella di origine
    video_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    for video_file in video_files:
        video_path = os.path.join(source_folder, video_file)
        extract_frames(video_path, output_folder)

# Percorsi delle cartelle
source_folder = r"C:\Users\Nico\Desktop\Dataset\Val\VideoFake"  # Cartella contenente i video
output_folder = r"C:\Users\Nico\Desktop\Dataset\FrameVal\FrameFake"  # Cartella di output per i frame estratti

# Processare i video
process_videos(source_folder, output_folder)
