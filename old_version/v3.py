import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Step 1: Preprocessing with Generator (from previous fixes)
def load_mri_scan(file_path, dtype=np.float32):
    """Load a NIfTI file and return the data array with specified dtype."""
    img = nib.load(file_path)
    return img.get_fdata(dtype=dtype)

def resize_volume(volume, target_shape=(128, 128, 128)):
    """Resize 3D volume to target shape using trilinear interpolation."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def z_score_normalize(volume):
    """Apply Z-score normalization."""
    mean = volume.mean()
    std = volume.std()
    return (volume - mean) / (std + 1e-8)

def preprocess_mri(t1_path, t2_path, flair_path, target_shape=(128, 128, 128)):
    """Preprocess T1, T2, FLAIR modalities."""
    t1 = load_mri_scan(t1_path, dtype=np.float32)
    t2 = load_mri_scan(t2_path, dtype=np.float32)
    flair = load_mri_scan(flair_path, dtype=np.float32)
    
    t1 = resize_volume(t1, target_shape)
    t2 = resize_volume(t2, target_shape)
    flair = resize_volume(flair, target_shape)
    
    t1 = z_score_normalize(t1)
    t2 = z_score_normalize(t2)
    flair = z_score_normalize(flair)
    
    stacked = np.stack([t1, t2, flair], axis=-1)
    return stacked

class BraTSGenerator(Sequence):
    """Data generator for BraTS 2021 dataset."""
    def __init__(self, data_dir, case_list, batch_size=2, target_shape=(128, 128, 128)):
        self.data_dir = data_dir
        self.case_list = case_list
        self.batch_size = batch_size
        self.target_shape = target_shape
    
    def __len__(self):
        return len(self.case_list) // self.batch_size
    
    def __getitem__(self, idx):
        batch_cases = self.case_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch, y_batch = [], []
        
        for case in batch_cases:
            case_path = os.path.join(self.data_dir, case)
            t1_path = os.path.join(case_path, f'{case}_t1.nii.gz')
            t2_path = os.path.join(case_path, f'{case}_t2.nii.gz')
            flair_path = os.path.join(case_path, f'{case}_flair.nii.gz')
            
            if all(os.path.exists(p) for p in [t1_path, t2_path, flair_path]):
                processed = preprocess_mri(t1_path, t2_path, flair_path, self.target_shape)
                X_batch.append(processed)
                
                seg_path = os.path.join(case_path, f'{case}_seg.nii.gz')
                label = 0
                if os.path.exists(seg_path):
                    seg = load_mri_scan(seg_path, dtype=np.float32)
                    label = 1 if np.any(seg > 0) else 0
                
                y_batch.append(label)
        
        return np.array(X_batch), np.array(y_batch)

# Step 2: Model Architecture (from previous fix)
class DepthwiseConv3D(layers.Layer):
    """Custom 3D Depthwise Convolution Layer."""
    def __init__(self, kernel_size, strides=(1, 1, 1), padding='same', use_bias=False, **kwargs):
        super(DepthwiseConv3D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape):
        self.filters = input_shape[-1]
        self.conv = layers.Conv3D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            groups=self.filters,
            use_bias=self.use_bias
        )
        super(DepthwiseConv3D, self).build(input_shape)

    def call(self, inputs):
        return self.conv(inputs)

    def get_config(self):
        config = super(DepthwiseConv3D, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias
        })
        return config

def conv3d_bn(x, filters, kernel_size, strides=(1, 1, 1), padding='same'):
    """3D Convolution + BatchNorm + ReLU."""
    x = layers.Conv3D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def depthwise_separable_conv3d(x, filters, kernel_size, strides=(1, 1, 1)):
    """3D Depthwise Separable Convolution."""
    x = DepthwiseConv3D(kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def se_block(x, ratio=4):
    """Squeeze-and-Excitation block."""
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling3D()(x)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 1, channels))(se)
    return layers.multiply([x, se])

def inverted_residual_block(x, filters, expansion, strides):
    """Inverted Residual Block with SE."""
    in_channels = x.shape[-1]
    x_exp = layers.Conv3D(expansion * in_channels, 1, padding='same', use_bias=False)(x)
    x_exp = layers.BatchNormalization()(x_exp)
    x_exp = layers.ReLU()(x_exp)
    x_exp = DepthwiseConv3D(3, strides=strides, padding='same', use_bias=False)(x_exp)
    x_exp = layers.BatchNormalization()(x_exp)
    x_exp = layers.ReLU()(x_exp)
    x_exp = layers.Conv3D(filters, 1, padding='same', use_bias=False)(x_exp)
    x_exp = layers.BatchNormalization()(x_exp)
    x_exp = se_block(x_exp)
    if strides == (1, 1, 1) and in_channels == filters:
        x = layers.add([x, x_exp])
    else:
        x = x_exp
    return x

def build_efficientnet_lite_3d(input_shape=(128, 128, 128, 3), num_classes=1):
    """Build EfficientNet-Lite 3D model."""
    inputs = layers.Input(shape=input_shape)
    x = conv3d_bn(inputs, 32, 3, strides=(2, 2, 2))
    x = inverted_residual_block(x, filters=16, expansion=1, strides=(1, 1, 1))
    x = inverted_residual_block(x, filters=24, expansion=6, strides=(2, 2, 2))
    x = inverted_residual_block(x, filters=24, expansion=6, strides=(1, 1, 1))
    x = inverted_residual_block(x, filters=40, expansion=6, strides=(2, 2, 2))
    x = inverted_residual_block(x, filters=40, expansion=6, strides=(1, 1, 1))
    x = inverted_residual_block(x, filters=80, expansion=6, strides=(2, 2, 2))
    x = conv3d_bn(x, 1280, 1)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    return models.Model(inputs, outputs)

# Create generators
data_dir = 'D:/ML PROJECTS/brain_tumor/dataset/BraTS2021_Training_Data'  # Update to your path
all_cases = [c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]
train_cases, temp_cases = train_test_split(all_cases, test_size=0.3, random_state=42)
val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)
batch_size = 1
train_generator = BraTSGenerator(data_dir, train_cases, batch_size)
val_generator = BraTSGenerator(data_dir, val_cases, batch_size)
test_generator = BraTSGenerator(data_dir, test_cases, batch_size)

# Build model
model = build_efficientnet_lite_3d()
model.summary()

# Step 3: Training Configuration
# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max')
]

# Train model with generators
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=callbacks
)

# Evaluate on test generator
test_results = model.evaluate(test_generator)
print(f"Test Accuracy: {test_results[1]:.4f}, Test AUC: {test_results[2]:.4f}")