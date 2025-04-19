import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Add, Activation, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, ResNet50
import os

def create_model(num_classes=20, model_type='custom_cnn', input_shape=(28, 28, 1)):
    """
    Create a CNN model for symbol recognition.
    
    Args:
        num_classes: Number of classes to predict
        model_type: Type of model to create ('custom_cnn', 'improved_cnn', 'mobilenet', 'resnet')
        input_shape: Input shape of images
        
    Returns:
        Compiled Keras model
    """
    if model_type == 'custom_cnn':
        model = Sequential([
            # First convolutional block
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same'),
            BatchNormalization(),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Second convolutional block
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Third convolutional block
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Flatten the output and feed it into dense layers
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
    
    elif model_type == 'improved_cnn':
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First block with residual connection
        x = Conv2D(64, kernel_size=(3, 3), padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection for first block
        shortcut = Conv2D(64, kernel_size=(1, 1), padding='same')(inputs)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # Second block with residual connection
        previous = x
        x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection for second block
        shortcut = Conv2D(128, kernel_size=(1, 1), padding='same')(previous)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # Third block with residual connection
        previous = x
        x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        
        # Residual connection for third block
        shortcut = Conv2D(256, kernel_size=(1, 1), padding='same')(previous)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.3)(x)
        
        # Global average pooling + dense layers
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        
        # Output layer
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
    elif model_type == 'mobilenet':
        # For transfer learning with MobileNetV2
        # Resize input to 96x96 for MobileNetV2
        base_model = MobileNetV2(weights='imagenet', include_top=False, 
                                input_shape=(96, 96, 3))
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create new model on top
        inputs = Input(shape=input_shape)
        
        # Preprocess: convert to 3 channels and resize to 96x96
        x = tf.keras.layers.Lambda(
            lambda x: tf.image.grayscale_to_rgb(tf.image.resize(x, [96, 96]))
        )(inputs)
        
        x = base_model(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
    elif model_type == 'resnet':
        # For transfer learning with ResNet50
        # Resize input to 224x224 for ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False, 
                             input_shape=(224, 224, 3))
        
        # Freeze the base model
        base_model.trainable = False
        
        # Create new model on top
        inputs = Input(shape=input_shape)
        
        # Preprocess: convert to 3 channels and resize to 224x224
        x = tf.keras.layers.Lambda(
            lambda x: tf.image.grayscale_to_rgb(tf.image.resize(x, [224, 224]))
        )(inputs)
        
        x = base_model(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_val, y_val, batch_size=32, epochs=20):
    """
    Train the model on the given data.
    
    Args:
        model: Keras model to train
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        batch_size: Batch size for training
        epochs: Number of epochs to train for
        
    Returns:
        Training history
    """
    # Data augmentation to improve model robustness
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Create a callback to save the best model
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/symbol_recognition_model/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=[checkpoint, early_stopping]
    )
    
    return history

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained Keras model
        x_test: Test data
        y_test: Test labels
        
    Returns:
        Test loss and accuracy
    """
    return model.evaluate(x_test, y_test)

def save_model(model, path='models/symbol_recognition_model'):
    """
    Save the model to disk.
    
    Args:
        model: Trained Keras model
        path: Path to save the model
    """
    model.save(path)
    print(f"Model saved to {path}") 