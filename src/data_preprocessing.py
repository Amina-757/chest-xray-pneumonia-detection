
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

def get_data_paths(base_path='/content/chest_xray'):
    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    test_dir = os.path.join(base_path, 'test')
    return train_dir, val_dir, test_dir

def create_data_generators(base_path='/content/chest_xray', batch_size=BATCH_SIZE):
    train_dir, val_dir, test_dir = get_data_paths(base_path)
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size, class_mode='binary',
        classes=CLASS_NAMES, shuffle=True, seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size, class_mode='binary',
        classes=CLASS_NAMES, shuffle=False, seed=42
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size, class_mode='binary',
        classes=CLASS_NAMES, shuffle=False, seed=42
    )
    
    return train_generator, val_generator, test_generator

def print_dataset_info(base_path='/content/chest_xray'):
    train_dir, val_dir, test_dir = get_data_paths(base_path)
    print("=" * 50)
    print("📊 DATASET INFORMATION")
    print("=" * 50)
    for split_name, split_dir in [('Train', train_dir), ('Validation', val_dir), ('Test', test_dir)]:
        if os.path.exists(split_dir):
            normal = len(os.listdir(os.path.join(split_dir, 'NORMAL')))
            pneumonia = len(os.listdir(os.path.join(split_dir, 'PNEUMONIA')))
            total = normal + pneumonia
            print(f"\n{split_name} Set:")
            print(f"  NORMAL: {normal} ({normal/total*100:.1f}%)")
            print(f"  PNEUMONIA: {pneumonia} ({pneumonia/total*100:.1f}%)")
            print(f"  Total: {total}")
    print("=" * 50)
