import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Define the ColorDetect class for color detection
class ColorDetect:
    def __init__(self, image):
        if isinstance(image, np.ndarray):
            self.image = image
        elif isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            raise TypeError("The image parameter accepts a numpy array or string file path only")

        self.color_description = {}

    def get_color_count(self, color_count=5):
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        reshape = rgb.reshape((rgb.shape[0] * rgb.shape[1], 3))
        cluster = KMeans(n_clusters=color_count).fit(reshape)
        unique_colors = self._find_unique_colors(cluster, cluster.cluster_centers_)

        for percentage, v in unique_colors.items():
            rgb_value = list(np.around(v))
            self.color_description[str(rgb_value)] = round(percentage, 2)

        return self.color_description

    def _find_unique_colors(self, cluster, centroids):
        labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
        hist, _ = np.histogram(cluster.labels_, bins=labels)
        hist = hist.astype("float")
        hist /= hist.sum()

        colors = sorted(
            [((percent * 100), color) for (percent, color) in zip(hist, centroids)]
        )
        return dict(colors)

# Define paths to your dataset
train_data_dir = r'C:\Users\hafsa\Downloads\CT CCP\dataset'
validation_data_dir = r'C:\Users\hafsa\Downloads\CT CCP\validation'

# Image parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 50

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['black', 'transparent', 'colorful'])

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    classes=['black', 'transparent', 'colorful'])

# Load the VGG16 model, excluding the top dense layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze the last few layers for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Build the model
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

# Learning rate scheduler and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping])

# Save the model
model.save('plastic_classifier_model.keras')

# Load the trained model
model = tf.keras.models.load_model('plastic_classifier_model.keras')
print("Model loaded")

# Dictionary to map predictions to conveyor belts
belt_map = {
    0: 'A - Black objects',
    1: 'B - Transparent objects',
    2: 'C - Colorful objects'
}

def classify_image_from_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read the frame from the camera.")
            break

        cv2.imshow('Camera Feed - Press "c" to capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            # Color detection
            color_detector = ColorDetect(frame)
            colors = color_detector.get_color_count(color_count=5)
            print(f"Dominant colors: {colors}")

            # Classification
            frame_resized = cv2.resize(frame, (img_width, img_height))
            frame_array = frame_resized.astype('float32') / 255.0
            frame_array = np.expand_dims(frame_array, axis=0)

            prediction = model.predict(frame_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            conveyor_belt = belt_map[predicted_class]

            print(f"Conveyor Belt: {conveyor_belt}")

            cv2.imshow('Captured Image', frame_resized)
            cv2.waitKey(0)
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Test the classifier with an image from the camera
classify_image_from_camera()