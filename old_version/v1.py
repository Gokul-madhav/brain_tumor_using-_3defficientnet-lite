import tarfile
import tempfile
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from typing import Dict, List, Optional
import time

# =====================
# 1. Data Loader with Dynamic Resizing
# =====================
class UniversalBraTSLoader:
    def __init__(self, tar_path: str, target_size: tuple = (128, 128, 128)):
        self.tar_path = tar_path
        self.target_size = target_size
        self.case_modality_map = self._build_file_map()
        
        if not self.case_modality_map:
            print("\n[DEBUG] Tar contents:")
            with tarfile.open(self.tar_path) as tar:
                print(tar.getnames())
            raise ValueError("No valid cases found - check naming conventions")

    def _build_file_map(self) -> Dict[str, Dict[str, str]]:
        file_map = {}
        with tarfile.open(self.tar_path) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.nii.gz'):
                    base = os.path.basename(member.name)
                    parts = base.split('_')
                    if len(parts) >= 3 and parts[-1].split('.')[0] in ['t1', 't2', 'flair']:
                        case_id = '_'.join(parts[:2])
                        mod = parts[-1].split('.')[0]
                        if case_id not in file_map:
                            file_map[case_id] = {}
                        file_map[case_id][mod] = member.name
        return file_map

    def load_case(self, case_id: str) -> np.ndarray:
        volumes = []
        missing_mods = []
        
        with tarfile.open(self.tar_path) as tar:
            for mod in ['t1', 't2', 'flair']:
                if mod in self.case_modality_map.get(case_id, {}):
                    try:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            tar.extract(self.case_modality_map[case_id][mod], path=tmpdir)
                            img = nib.load(os.path.join(tmpdir, self.case_modality_map[case_id][mod])).get_fdata()
                            img = self._preprocess_volume(img)
                            volumes.append(img)
                    except Exception as e:
                        print(f"Error loading {mod} for {case_id}: {str(e)}")
                        volumes.append(np.zeros(self.target_size, dtype=np.float32))
                        missing_mods.append(mod)
                else:
                    volumes.append(np.zeros(self.target_size, dtype=np.float32))
                    missing_mods.append(mod)
        
        if missing_mods:
            print(f"Warning: Case {case_id} missing modalities: {', '.join(missing_mods)}")
        return np.stack(volumes, axis=-1)

    def _preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        volume = volume.astype(np.float32)
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-7)
        
        # Dynamic resizing for any input dimensions
        if volume.shape != self.target_size:
            # print(f"Resizing volume from {volume.shape} to {self.target_size}")
            # Resize each slice independently
            resized_slices = []
            for i in range(volume.shape[2]):  # Assuming z-axis is last
                slice_2d = volume[:, :, i]
                resized_slice = tf.image.resize(
                    np.expand_dims(slice_2d, axis=-1),
                    self.target_size[:2]
                ).numpy()[..., 0]
                resized_slices.append(resized_slice)
            
            # Stack slices and adjust depth
            volume = np.stack(resized_slices, axis=2)
            if volume.shape[2] > self.target_size[2]:  # If too deep
                volume = volume[:, :, :self.target_size[2]]
            elif volume.shape[2] < self.target_size[2]:  # If not deep enough
                padding = [(0,0), (0,0), (0, self.target_size[2] - volume.shape[2])]
                volume = np.pad(volume, padding)
        
        return volume

# =====================
# 2. Flexible 3D CNN Model
# =====================
def build_flexible_3d_cnn(input_shape=(128, 128, 128, 3), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Stem with dynamic shape adaptation
    x = layers.Conv3D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Middle layers
    x = layers.Conv3D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(2)(x)
    
    x = layers.Conv3D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling3D(2)(x)
    
    # Head with global pooling
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    return tf.keras.Model(inputs, outputs)

# =====================
# 3. Main Pipeline
# =====================
def main():
    # Configuration
    brats_tar_path = "D:/ML PROJECTS/brain_tumor/dataset/BraTS2021_Training_Data.tar"
    target_size = (128, 128, 128)  # Target dimensions after resizing
    epochs = 5
    
    try:
        # Initialize loader
        print("Initializing data loader...")
        loader = UniversalBraTSLoader(brats_tar_path, target_size=target_size)
        
        # Get cases
        case_ids = list(loader.case_modality_map.keys())
        print(f"\nLoaded {len(case_ids)} cases: {case_ids}")
        
        # Load sample to check dimensions
        sample = loader.load_case(case_ids[0])
        print(f"\nSample shape after loading: {sample.shape}")
        print(f"Data range: {np.min(sample):.2f} to {np.max(sample):.2f}")
        
        # Handle single-case scenario
        if len(case_ids) == 1:
            print("\n[WARNING] Single-case mode - using synthetic validation data")
            real_vol = sample
            fake_vol = np.random.normal(size=real_vol.shape).astype(np.float32)
            
            train_data = np.stack([real_vol])
            train_labels = np.array([1])
            val_data = np.stack([fake_vol])
            val_labels = np.array([0])
        else:
            labels = np.ones(len(case_ids))
            train_ids, val_ids, train_labels, val_labels = train_test_split(
                case_ids, labels, test_size=0.2, random_state=42
            )
            train_data = np.stack([loader.load_case(cid) for cid in train_ids])
            val_data = np.stack([loader.load_case(cid) for cid in val_ids])
        
        # Build model
        print("\nBuilding model...")
        model = build_flexible_3d_cnn(input_shape=(*target_size, 3))
        model.compile(
            optimizer=optimizers.Adam(3e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        model.summary()
        
        # Train
        print("\nStarting training...")
        history = model.fit(
            x=train_data, y=train_labels,
            validation_data=(val_data, val_labels),
            epochs=epochs,
            batch_size=2,
            callbacks=[
                callbacks.ModelCheckpoint(
                    'best_model.keras',
                    monitor='val_auc',
                    save_best_only=True,
                    mode='max'
                ),
                callbacks.EarlyStopping(
                    patience=3,
                    monitor='val_auc',
                    mode='max'
                )
            ]
        )
        
        # Benchmark
        start_time = time.time()
        _ = model.predict(sample[np.newaxis, ...])
        print(f"\nInference time: {(time.time()-start_time)*1000:.2f}ms")
        
        # Save final model
        model.save('brain_tumor_classifier.keras')
        print("\nTraining complete! Models saved:")
        print("- best_model.keras (best validation)")
        print("- brain_tumor_classifier.keras (final model)")
        
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("\nDebugging steps:")
        print("1. Verify tar contains files like: BraTS2021_00000_t1.nii.gz")
        print("2. Check file exists at:", brats_tar_path)
        print("3. The case must have all 3 modalities (t1, t2, flair)")

if __name__ == "__main__":
    main()