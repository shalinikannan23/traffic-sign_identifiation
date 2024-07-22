import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Dummy data generator for training
def create_dummy_data(num_samples, img_size, num_classes):
    X = np.random.rand(num_samples, img_size, img_size, 1)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y

# Define your model
cnn_model = Sequential()
cnn_model.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(32,32,1)))
cnn_model.add(AveragePooling2D(2,2))
cnn_model.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
cnn_model.add(AveragePooling2D(2,2))
cnn_model.add(Flatten())
cnn_model.add(Dense(units=120, activation='relu'))
cnn_model.add(Dense(units=84, activation='relu'))
cnn_model.add(Dense(units=43, activation='softmax'))
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create dummy training data
X_train, y_train = create_dummy_data(1000, 32, 43)

# Train the model on dummy data
cnn_model.fit(X_train, y_train, epochs=5)

# Save the model architecture to JSON
model_json = cnn_model.to_json()
with open("C:/Users/SEC/Downloads/model_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Define and save class names using pickle
class_names = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons', 11: 'Right-of-way at the next intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End of no passing by vehicles over 3.5 metric tons'
}

with open("C:/Users/SEC/Downloads/class_names.pkl", "wb") as class_file:
    pickle.dump(class_names, class_file)

# Load the model architecture from JSON
with open("C:/Users/SEC/Downloads/model_architecture.json", "r") as json_file:
    model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(model_json)

# Load the class names using pickle
with open("C:/Users/SEC/Downloads/class_names.pkl", "rb") as class_file:
    loaded_class_names = pickle.load(class_file)

# Verify the loaded model and class names
print(loaded_model.summary())
print(loaded_class_names)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32), color_mode='grayscale')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = (img - 128) / 128  # Normalize the image
    return img

def get_class_name(class_id):
    return loaded_class_names.get(class_id, "Unknown Class")

def predict_traffic_sign_class(img_path):
    img = preprocess_image(img_path)
    prediction = loaded_model.predict(img)
    class_id = np.argmax(prediction, axis=1)[0]
    class_name = get_class_name(class_id)
    return class_name

def plot_image_with_prediction(img_path):
    class_name = predict_traffic_sign_class(img_path)
    plt.imshow(image.load_img(img_path, target_size=(32, 32)), cmap='gray')
    plt.title(f"Predicted: {class_name}")
    plt.show()

# Example usage
img_path = "C:/Users/SEC/Downloads/Screenshot 2024-07-20 111906.png" # Replace with the path to your image
predicted_class = predict_traffic_sign_class(img_path)
print(f"Predicted Class: {predicted_class}")
plot_image_with_prediction(img_path)
