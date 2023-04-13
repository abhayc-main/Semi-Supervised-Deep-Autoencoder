import tensorflow as tf
import numpy as np

# Original Version of the code kept for reference

# Issue with this code are the functional approach

# Data preprocessing function
def preprocess_data(data):
    # Extract relevant features
    features = extract_features(data)

    # Normalize features
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    features = (features - mean) / std

    # Split into training and testing sets
    train_data, test_data, train_labels, test_labels = split_data(
        features, labels)

    return train_data, test_data, train_labels, test_labels


# Anomaly detection module
def build_autoencoder(input_dim, latent_dim):
    # Build encoder network
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoder = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    encoder = tf.keras.layers.Dense(32, activation='relu')(encoder)
    encoder = tf.keras.layers.Dense(latent_dim, activation='relu')(encoder)

    # Build decoder network
    decoder = tf.keras.layers.Dense(32, activation='relu')(encoder)
    decoder = tf.keras.layers.Dense(64, activation='relu')(decoder)
    decoder = tf.keras.layers.Dense(input_dim, activation='linear')(decoder)

    # Combine encoder and decoder into autoencoder
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
    return autoencoder


def train_autoencoder(autoencoder, train_data):
    # Train autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train_data, train_data, epochs=10,
                    batch_size=128, verbose=1)


def detect_anomalies(autoencoder, test_data):
    # Detect anomalies using reconstruction error
    reconstructions = autoencoder.predict(test_data)
    reconstruction_error = np.mean(np.abs(reconstructions - test_data), axis=1)
    threshold = np.mean(reconstruction_error) + np.std(reconstruction_error)
    labels = np.where(reconstruction_error > threshold, 1, 0)
    return labels


# Classification module
def build_model(input_shape, num_classes):
    # Build model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(model, train_data, train_labels):
    # Train model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=10, batch_size=128, verbose=1)


def predict_classes(model, test_data):
    # Predict classes
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels


# Ensembling module
def majority_voting(labels1, labels2):
    # Combine labels using majority voting
    combined_labels = np.where((labels1 + labels2) > 1, 1, 0)
    return combined_labels


# Split data into train and test sets
train_set, test_set = train_test_split(data, test_size=0.2)

# Preprocess data
scaler = StandardScaler()
train_preprocessed = preprocess_data(train_set, scaler)
test_preprocessed = preprocess_data(test_set, scaler)

# Build anomaly detection model
anomaly_detection_model = build_anomaly_detection_model(train_preprocessed)

# Detect anomalies in test data
anomalies = detect_anomalies(anomaly_detection_model, test_preprocessed)

# Build classification model
classification_model = build_classification_model(train_preprocessed)

# Evaluate classification model on test data
test_labels = test_set['label']
predictions = classify_data(classification_model, test_preprocessed)
accuracy = evaluate_classification(predictions, test_labels)

# Combine outputs using ensembling module
combined_predictions = ensemble(anomalies, predictions)

# Evaluate combined predictions on test data
combined_accuracy = evaluate_classification(combined_predictions, test_labels)
