import tensorflow as tf
import numpy as np
import os
import struct
import glob

def load_spectrograms_from_bin(directory):
    """Load spectrograms from .bin files created by Red Pitaya"""
    spectrograms = []
    filenames = []

    bin_files = sorted(glob.glob(f"{directory}/stft_*.bin"))
    print(f"Found {len(bin_files)} spectrogram files")

    for file in bin_files:
        try:
            with open(file, 'rb') as f:
                # Read metadata (7 uint32 values)
                meta = struct.unpack('7I', f.read(7 * 4))
                samples_20ms, nperseg, noverlap, num_subwindows, new_freq_bins,$

                print(f"File {os.path.basename(file)}: {new_freq_bins}x{num_sub$

                # Skip the raw time-domain data
                f.seek(samples_20ms * 4, 1)  # Skip float32 time data

                # Read the spectrogram data
                spec_size = new_freq_bins * num_subwindows
                spec_data = struct.unpack(f'{spec_size}f',
f.read(spec_size * 4))
                spec_array = np.array(spec_data).reshape(new_freq_bins,
num_subwindows)

                spectrograms.append(spec_array)
                filenames.append(os.path.basename(file))

        except Exception as e:
            print(f"Error loading {file}: {e}")

    if len(spectrograms) > 0:
        return np.array(spectrograms), filenames
    else:
        return np.array([]), []

def create_dummy_labels(num_samples):
    """Create dummy labels for testing - replace with real labels later"""
    # For now, create random binary labels
    return np.random.randint(0, 2, num_samples)

def create_cnn_model(input_shape):
    """Create CNN architecture - adjust this to match your C
implementation"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        # Add channel dimension if needed
        tf.keras.layers.Reshape(input_shape + (1,)),
    
        # Convolutional layers
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),  
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
            
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
        
    return model
        
def export_weights_for_c(model, filename='weights_for_c.bin'):
    """Export model weights in a format that C can read"""
    weights = model.get_weights()
        
    with open(filename, 'wb') as f:
        # Write number of layers
        f.write(struct.pack('I', len(weights)))
        
        for i, layer_weights in enumerate(weights):
            # Write layer info
            shape = layer_weights.shape
            f.write(struct.pack('I', len(shape)))  # Number of dimensions

            for dim in shape:
                f.write(struct.pack('I', dim))  # Each dimension size
        
            # Write the actual weights as float32
            layer_weights.astype(np.float32).tofile(f)
            
            print(f"Layer {i}: shape {shape}, {layer_weights.size} weights")
            
      print(f"Weights exported to {filename}")
        
# Main training workflow
if __name__ == "__main__":
    # Set up paths
    spectrogram_dir = os.path.expanduser("~/Desktop/SPECTROGRAMS")
    
    print("=== CNN Training Workflow ===")
    print(f"Loading spectrograms from: {spectrogram_dir}")
        
    # Load spectrograms
    X, filenames = load_spectrograms_from_bin(spectrogram_dir)
        
    if len(X) == 0:
        print("No spectrograms found! Make sure you've run the sync script.")
        print("Expected files like: stft_000000.bin, stft_000001.bin, etc.")
        exit(1)
            
    print(f"Loaded {len(X)} spectrograms with shape {X.shape}")
        
    # Create dummy labels (replace with real labels when you have bubble chambe$
    y = create_dummy_labels(len(X))
    print(f"Created {len(y)} dummy labels")
            
    # Create and compile model
    input_shape = X.shape[1:]  # Remove batch dimension
    model = create_cnn_model(input_shape)

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== Model Architecture ===")
    model.summary()
    
    # Split data for training/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
        
    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} sa$
    
    # Train the model
    print("\n=== Training ===")
    history = model.fit(
       X_train, y_train,
        epochs=10,
        batch_size=8,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    # Save weights in Keras format
    model.save_weights('trained_weights.weights.h5')
    print("\nKeras weights saved to: trained_weights.h5")
     
    # Export weights for C implementation
    export_weights_for_c(model, 'weights_for_c.bin')
    
    print("\n=== Training Complete ===")
    print("Next steps:")
    print("1. Copy weights_for_c.bin to your Red Pitaya")
    print("2. Modify your C code to load these weights")
    print("3. Test the trained model on new data")
