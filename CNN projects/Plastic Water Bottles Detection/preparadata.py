import os
import shutil
import random
from ultralytics import YOLO
from PIL import Image
#---------------------------------------------------------------
def copia_ficheiros(base_path,images_path,labels_path,files, set_type):
    for file in files: # Copy image
        shutil.copy(os.path.join(images_path, file), os.path.join(base_path, set_type, 'images')) # Copy corresponding label
        label_file = file.rsplit('.', 1)[0] + '.txt'
        shutil.copy(os.path.join(labels_path, label_file), os.path.join(base_path, set_type, 'labels'))
#---------------------------------------------------------------     
def preparadataset(base_path):
    random.seed(42)
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    for set_type in ['train', 'val', 'test']:
        for content_type in ['images', 'labels']:
            os.makedirs(os.path.join(base_path, set_type, content_type), exist_ok=True)

    all_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    random.shuffle(all_files)

    total_files = len(all_files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

    copia_ficheiros(base_path,images_path,labels_path,train_files, 'train')
    copia_ficheiros(base_path,images_path,labels_path,val_files, 'val')
    copia_ficheiros(base_path,images_path,labels_path,test_files, 'test')

preparadataset("C:\\Users\\madal\\OneDrive\\Ambiente de Trabalho\\GitHub Computer Vision\\OpenCV Projects\\CNN projects\\Plastic Water Bottles Detection\\train")

