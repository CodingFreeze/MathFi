import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import json

from app.models.cnn_model import create_model, train_model, evaluate_model, save_model
from app.utils.data_preprocessing import prepare_dataset

def main():
    print("Preparing the dataset...")
    x_train, y_train, x_test, y_test = prepare_dataset()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create the model
    print("Creating the model...")
    num_classes = 20  # 10 digits (0-9) + 10 math symbols (+, -, ×, ÷, =, x, y, (, ), .)
    model = create_model(num_classes=num_classes)
    
    # Print model summary
    model.summary()
    
    # Create label map for inference (will be saved with the model)
    label_map = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
        5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '+', 11: '-', 12: '×', 13: '÷', 14: '=',
        15: 'x', 16: 'y', 17: '(', 18: ')', 19: '.'
    }
    
    # Ensure the model directory exists
    model_dir = 'models/symbol_recognition_model'
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the label map
    with open(os.path.join(model_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)
    
    # Create data augmentation for training
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Create callbacks
    callbacks = [
        # Save the best model
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate when plateauing
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    print("Training the model...")
    batch_size = 32
    epochs = 20  # Use more epochs for production, fewer for testing
    
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate the model
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"Test accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    # Save the final model (we also saved the best model during training)
    print("Saving the model...")
    save_model(model, os.path.join(model_dir, 'final_model'))
    
    print("Done!")

if __name__ == "__main__":
    main() 