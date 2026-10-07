import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import math
import logging
import gc
try:
    import tf_keras
except ImportError:
    print("Error: tf_keras not installed. Run 'pip install tf_keras'")
    exit(1)
try:
    import tensorflow_model_optimization as tfmot
except ImportError:
    print("Error: tensorflow-model-optimization not installed. Run 'pip install tensorflow-model-optimization'")
    exit(1)
try:
    import tf2onnx
except ImportError:
    print("Warning: tf2onnx not installed. ONNX conversion will fail. Run 'pip install tf2onnx'.")

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
if int(tf.__version__.split('.')[0]) < 2 or (int(tf.__version__.split('.')[0]) == 2 and int(tf.__version__.split('.')[1]) < 17):
    print("Warning: TensorFlow version < 2.17 may cause compatibility issues.")

# Setup logging
logging.basicConfig(filename='training.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Step 1: Preprocessing with Generator
def load_mri_scan(file_path, dtype=np.float32):
    try:
        img = nib.load(file_path)
        data = img.get_fdata(dtype=dtype)
        if not np.isfinite(data).all():
            logging.warning(f"Non-finite values in {file_path}")
            return None
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def resize_volume(volume, target_shape=(128, 128, 128)):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)

def z_score_normalize(volume):
    mean = volume.mean()
    std = volume.std()
    if std < 1e-6:
        logging.warning("Low variance in volume, returning unnormalized")
        return volume
    normalized = (volume - mean) / std
    if not np.isfinite(normalized).all():
        logging.warning("Non-finite values after normalization")
        return volume
    return normalized

def preprocess_mri(t1_path, t2_path, flair_path, target_shape=(128, 128, 128)):
    t1 = load_mri_scan(t1_path, dtype=np.float32)
    t2 = load_mri_scan(t2_path, dtype=np.float32)
    flair = load_mri_scan(flair_path, dtype=np.float32)
    
    if any(x is None for x in [t1, t2, flair]):
        logging.error(f"Invalid modalities: {t1_path}, {t2_path}, {flair_path}")
        return None
    
    t1 = resize_volume(t1, target_shape)
    t2 = resize_volume(t2, target_shape)
    flair = resize_volume(flair, target_shape)
    
    t1 = z_score_normalize(t1)
    t2 = z_score_normalize(t2)
    flair = z_score_normalize(flair)
    
    stacked = np.stack([t1, t2, flair], axis=-1)
    if np.allclose(stacked, stacked[0, 0, 0, :], rtol=1e-3, atol=1e-3):
        logging.warning("Uniform values in stacked volume")
        return None
    
    logging.info(f"Processed data range: min={stacked.min():.4f}, max={stacked.max():.4f}, std={stacked.std():.4f}")
    # print(f"Processed data range: min={stacked.min():.4f}, max={stacked.max():.4f}, std={stacked.std():.4f}")
    return stacked

class BraTSGenerator(Sequence):
    def __init__(self, data_dir, case_list, batch_size=2, target_shape=(128, 128, 128), neg_ratio=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.target_shape = target_shape
        self.neg_ratio = neg_ratio
        self.skipped_cases = []
        self.valid_cases = []
        tumor_sizes = []
        for case in case_list:
            case_path = os.path.join(self.data_dir, case)
            t1_path = os.path.join(case_path, f'{case}_t1.nii.gz')
            t2_path = os.path.join(case_path, f'{case}_t2.nii.gz')
            flair_path = os.path.join(case_path, f'{case}_flair.nii.gz')
            seg_path = os.path.join(case_path, f'{case}_seg.nii.gz')
            if all(os.path.exists(p) for p in [t1_path, t2_path, flair_path, seg_path]):
                processed = preprocess_mri(t1_path, t2_path, flair_path, target_shape)
                if processed is not None:
                    self.valid_cases.append(case)
                    size = 0
                    seg = load_mri_scan(seg_path, dtype=np.float32)
                    if seg is not None:
                        size = np.sum(seg > 0)
                    tumor_sizes.append((case, size))
                else:
                    logging.warning(f"Skipping case {case} during init: preprocessing failure")
                    self.skipped_cases.append(case)
            else:
                logging.warning(f"Skipping case {case} during init: missing files")
                self.skipped_cases.append(case)
        
        if not self.valid_cases:
            logging.error("No valid cases found")
            print("Error: No valid cases found")
            exit(1)
        
        tumor_sizes.sort(key=lambda x: x[1])
        neg_count = max(1, int(len(self.valid_cases) * neg_ratio))
        self.labels = {case: 0 if i < neg_count else 1 for i, (case, _) in enumerate(tumor_sizes)}
        self.pos_cases = [case for case in self.valid_cases if self.labels[case] == 1]
        self.neg_cases = [case for case in self.valid_cases if self.labels[case] == 0]
        print(f"Generator initialized with {len(self.valid_cases)} valid cases: "
              f"{len(self.pos_cases)} positive, {len(self.neg_cases)} negative")
        print(f"Skipped during init: {len(self.skipped_cases)} cases")
        logging.info(f"Generator: {len(self.valid_cases)} valid, {len(self.pos_cases)} pos, {len(self.neg_cases)} neg")

    def __len__(self):
        return max(1, math.ceil(len(self.valid_cases) / self.batch_size))
    
    def __getitem__(self, idx):
        X_batch, y_batch = [], []
        neg_per_batch = 1 if self.neg_cases else 0
        pos_per_batch = self.batch_size - neg_per_batch
        
        pos_indices = np.random.choice(len(self.pos_cases), pos_per_batch, replace=len(self.pos_cases) < pos_per_batch)
        neg_indices = np.random.choice(len(self.neg_cases), neg_per_batch, replace=len(self.neg_cases) < neg_per_batch) if neg_per_batch else []
        batch_cases = [self.pos_cases[i] for i in pos_indices]
        batch_cases += [self.neg_cases[i] for i in neg_indices]
        np.random.shuffle(batch_cases)
        
        for case in batch_cases:
            case_path = os.path.join(self.data_dir, case)
            t1_path = os.path.join(case_path, f'{case}_t1.nii.gz')
            t2_path = os.path.join(case_path, f'{case}_t2.nii.gz')
            flair_path = os.path.join(case_path, f'{case}_flair.nii.gz')
            
            processed = preprocess_mri(t1_path, t2_path, flair_path, self.target_shape)
            if processed is not None:
                X_batch.append(processed)
                y_batch.append(self.labels[case])
            else:
                logging.warning(f"Skipping case {case} during batch: preprocessing failure")
                self.skipped_cases.append(case)
        
        while len(X_batch) < self.batch_size:
            case = self.neg_cases[np.random.randint(len(self.neg_cases))] if np.random.rand() < 0.5 and self.neg_cases else self.pos_cases[np.random.randint(len(self.pos_cases))]
            case_path = os.path.join(self.data_dir, case)
            processed = preprocess_mri(
                os.path.join(case_path, f'{case}_t1.nii.gz'),
                os.path.join(case_path, f'{case}_t2.nii.gz'),
                os.path.join(case_path, f'{case}_flair.nii.gz'),
                self.target_shape
            )
            if processed is not None:
                X_batch.append(processed)
                y_batch.append(self.labels[case])
            else:
                X_batch.append(np.zeros(self.target_shape + (3,), dtype=np.float32))
                y_batch.append(0.0)
        
        X_batch = X_batch[:self.batch_size]
        y_batch = y_batch[:self.batch_size]
        logging.info(f"Batch {idx} labels: {y_batch}")
        # print(f"Batch {idx} labels: {y_batch}")
        gc.collect()
        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

def create_tf_dataset(generator):
    output_signature = (
        tf.TensorSpec(shape=(None, 128, 128, 128, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32)
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, 128, 128, 128, 3], [None])
    ).prefetch(tf.data.AUTOTUNE)
    return dataset

# Step 2: Model Architecture
class DepthwiseConv3D(layers.Layer):
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
    x = layers.Conv3D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def se_block(x, ratio=4):
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling3D()(x)
    se = layers.Dense(channels // ratio, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, 1, channels))(se)
    return layers.multiply([x, se])

def inverted_residual_block(x, filters, expansion, strides):
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
data_dir = 'D:/ML PROJECTS/brain_tumor/dataset/BraTS2021_Training_Data'
all_cases = [c for c in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, c))]
train_cases, temp_cases = train_test_split(all_cases, test_size=0.3, random_state=42)
val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)
batch_size = 2
train_generator = BraTSGenerator(data_dir, train_cases, batch_size, neg_ratio=0.2)
val_generator = BraTSGenerator(data_dir, val_cases, batch_size, neg_ratio=0.2)
test_generator = BraTSGenerator(data_dir, test_cases, batch_size, neg_ratio=0.2)

# Convert to tf.data.Dataset
train_dataset = create_tf_dataset(train_generator)
val_dataset = create_tf_dataset(val_generator)
test_dataset = create_tf_dataset(test_generator)

# Check label distribution
logging.info(f"Training labels: {sum(1 for l in train_generator.labels.values() if l == 1)} pos, "
             f"{sum(1 for l in train_generator.labels.values() if l == 0)} neg")
logging.info(f"Validation labels: {sum(1 for l in val_generator.labels.values() if l == 1)} pos, "
             f"{sum(1 for l in val_generator.labels.values() if l == 0)} neg")
logging.info(f"Test labels: {sum(1 for l in test_generator.labels.values() if l == 1)} pos, "
             f"{sum(1 for l in test_generator.labels.values() if l == 0)} neg")
print(f"Training labels: {sum(1 for l in train_generator.labels.values() if l == 1)} positive, "
      f"{sum(1 for l in train_generator.labels.values() if l == 0)} negative")
print(f"Validation labels: {sum(1 for l in val_generator.labels.values() if l == 1)} positive, "
      f"{sum(1 for l in val_generator.labels.values() if l == 0)} negative")
print(f"Test labels: {sum(1 for l in test_generator.labels.values() if l == 1)} positive, "
      f"{sum(1 for l in test_generator.labels.values() if l == 0)} negative")

# Debug dataset output
try:
    for X_batch, y_batch in train_dataset.take(1):
        logging.info(f"Sample batch shape: X={X_batch.shape}, y={y_batch.shape}")
        logging.info(f"Sample labels: {y_batch.numpy()}")
        logging.info(f"Sample data range: min={X_batch.numpy().min()}, max={X_batch.numpy().max()}")
        print(f"Sample batch shape: X={X_batch.shape}, y={y_batch.shape}")
        print(f"Sample labels: {y_batch.numpy()}")
        print(f"Sample data range: min={X_batch.numpy().min()}, max={X_batch.numpy().max()}")
except Exception as e:
    logging.error(f"Error fetching sample batch: {e}")
    print(f"Error fetching sample batch: {e}")
    exit(1)

# Build model
model = build_efficientnet_lite_3d()
model.summary()

# Step 3: Training Configuration
class_weights = {0: 1.0, 1: 1.0}
neg = sum(1 for l in train_generator.labels.values() if l == 0)
pos = sum(1 for l in train_generator.labels.values() if l == 1)
if neg > 0:
    class_weights = {0: (pos + neg) / (2.0 * neg), 1: (pos + neg) / (2.0 * pos)}
    logging.info(f"Class weights: {class_weights}")
    print(f"Class weights: {class_weights}")
else:
    logging.warning("No negative cases. Training may not generalize.")
    print("Warning: No negative cases. Training may not generalize.")
    exit(1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc', num_thresholds=200)]
)

callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=5, mode='max'),
    ModelCheckpoint('best_model.keras', monitor='val_auc', mode='max', save_best_only=True)
]

# Debug predictions
try:
    for X_batch, y_batch in train_dataset.take(1):
        preds = model.predict(X_batch)
        logging.info(f"Sample predictions before training: {preds.flatten()}, Labels: {y_batch.numpy()}")
        print(f"Sample predictions before training: {preds.flatten()}, Labels: {y_batch.numpy()}")
except Exception as e:
    logging.error(f"Error predicting on sample batch: {e}")
    print(f"Error predicting on sample batch: {e}")
    exit(1)

# Train model
logging.info("Starting training...")
try:
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        class_weight=class_weights,
        callbacks=callbacks,
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator),
        verbose=1
    )
    logging.info("Training completed")
except Exception as e:
    logging.error(f"Training failed: {e}")
    print(f"Training failed: {e}")
    raise
finally:
    gc.collect()
    tf.keras.backend.clear_session()

# Log skipped cases
logging.info(f"Skipped cases during training: {len(train_generator.skipped_cases)}")
if train_generator.skipped_cases:
    logging.info(f"First few skipped: {train_generator.skipped_cases[:5]}")
print(f"Skipped cases during training: {len(train_generator.skipped_cases)}")
if train_generator.skipped_cases:
    print(f"First few skipped: {train_generator.skipped_cases[:5]}")

test_results = model.evaluate(test_dataset, steps=len(test_generator))
logging.info(f"Test Accuracy: {test_results[1]:.4f}, Test AUC: {test_results[2]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}, Test AUC: {test_results[2]:.4f}")

# Step 4: Model Optimization
# 4.1 Pruning
try:
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=2000,
            end_step=10000
        )
    }
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    pruned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    pruned_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=2,
        class_weight=class_weights,
        callbacks=[
            tfmot.sparsity.keras.UpdatePruningStep(),
            EarlyStopping(monitor='val_auc', patience=10, mode='max')
        ],
        steps_per_epoch=len(train_generator),
        validation_steps=len(val_generator)
    )
    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    pruned_model.save('pruned_model.keras')
    logging.info("Pruned model saved as 'pruned_model.keras'")
    print("Pruned model saved as 'pruned_model.keras'")
except Exception as e:
    logging.error(f"Pruning failed: {e}")
    print(f"Pruning failed: {e}")

# 4.2 Quantization
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    def representative_dataset():
        for i in range(10):
            X_batch, _ = train_generator[i % len(train_generator)]
            yield [X_batch.astype(np.float32)]
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()
    with open('quantized_model.tflite', 'wb') as f:
        f.write(tflite_model)
    logging.info("Quantized model saved as 'quantized_model.tflite'")
    print("Quantized model saved as 'quantized_model.tflite'")
except Exception as e:
    logging.error(f"Quantization failed: {e}")
    print(f"Quantization failed: {e}")

# 4.3 TensorRT Optimization
try:
    onnx_model, _ = tf2onnx.convert.from_keras_model(pruned_model, output_path='model.onnx')
    logging.info("ONNX model saved as 'model.onnx'")
    print("ONNX model saved as 'model.onnx'")
    print("Run 'trtexec --onnx=model.onnx --saveEngine=model.trt --int8' to create TensorRT engine")
except Exception as e:
    logging.error(f"ONNX conversion failed: {e}")
    print(f"ONNX conversion failed: {e}")