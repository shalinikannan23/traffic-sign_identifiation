import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
import pickle
import json
from streamlit_option_menu import option_menu
from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Set page configuration
st.set_page_config(page_title="TRAFFIC SIGN IDENTIFICATION",
                   layout="wide",
                   page_icon="heart-pulse-fill")

# Load or initialize user database
def load_user_db():
    try:
        with open("user_db.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_user_db(user_db):
    with open("user_db.json", "w") as file:
        json.dump(user_db, file)

user_db = load_user_db()

# Load and display the logo
image = Image.open("logo.png")
col1, col2 = st.columns([0.4, 0.5])
with col1:
    st.image(image, width=350)

# Define and display the HTML title
html_title = """
    <style>
    .title-test {
    font-weight:bold;
    font-style:Cambria;
    padding:7px;
    border-radius:6px;
    }
    </style>
    <h1 class="title-test"> Navigate Safely</h1>
    <h4 class="title-test"> Recognize, React, and Drive Smart...</h4>"""
with col2:
    st.markdown(html_title, unsafe_allow_html=True)

st.markdown("""---""")

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Main Menu',
                           ['Home', 'Prediction', 'About'],
                           menu_icon='hospital-fill',
                           icons=['house-fill', 'activity', 'clipboard2-data-fill'],
                           default_index=0)

# Load the model architecture from JSON
with open("model_architecture.json", "r") as json_file:
    model_json = json_file.read()
loaded_model = tf.keras.models.model_from_json(model_json)

# Load the class names using pickle
with open("class_names.pkl", "rb") as class_file:
    loaded_class_names = pickle.load(class_file)

# Define preprocessing function
def preprocess_image(img):
    img = img.resize((32, 32)).convert('L')  # Resize and convert to grayscale
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=(0, -1))  # Add batch and channel dimensions
    img_array = (img_array - 128) / 128  # Normalize
    return img_array

def get_class_name(class_id):
    return loaded_class_names.get(class_id, "Unknown Class")

def predict_image(img):
    preprocessed_img = preprocess_image(img)
    prediction = loaded_model.predict(preprocessed_img)
    class_id = np.argmax(prediction, axis=1)[0]
    class_name = get_class_name(class_id)
    return class_name

def draw_bounding_box(img, box, label):
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline="red", width=5)  # Increased width
    draw.text((box[0], box[1]), label, fill="red")
    return img

def detect_objects(img):
    """
    Dummy object detection function.
    Replace this with your actual object detection model logic.
    """
    detected_boxes = [[50, 50, 150, 150]]  # Example box
    return detected_boxes

def main():
    # Set background color and style
    st.markdown(
        """
        <style>
        /* Apply background color to main content and sidebar */
        .reportview-container {
            background-color: #ffd9e4; /* Pastel pink background color */
        }
        .sidebar .sidebar-content {
            background-color: #ffd9e4; /* Match sidebar background */
        }

        /* Apply background color to header and footer */
        .css-1v3fvcr { 
            background-color: #ffd9e4 !important; /* Header background color */
        }
        .css-1bp8m5e { 
            background-color: #ffd9e4 !important; /* Footer background color */
        }

        /* Style the title */
        .css-18e3th9 {
            color: #d6336c; /* Title color */
            font-size: 2rem; /* Title size */
        }

        /* Style buttons */
        .css-1emrehy {
            background-color: #d6336c !important; /* Button background color */
            color: #fff !important; /* Button text color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Home Page
    if selected == "Home":
        st.title("Welcome to Traffic Sign Identifier")
        st.write("Your advanced tool for recognizing and interpreting traffic signs to enhance autonomous vehicle navigation.")
        st.image("giff.gif", use_column_width=True)

        # Buttons for login and signup
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                st.session_state.signup = False
        with col2:
            if st.button("Signup"):
                st.session_state.signup = True

        if not st.session_state.get("signup", False):
            st.subheader("Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Submit Login"):
                if username in user_db:
                    hashed_password = user_db[username]
                    if pwd_context.verify(password, hashed_password):
                        st.session_state["authenticated"] = True
                        st.success("Login successful!")
                    else:
                        st.error("Invalid password")
                else:
                    st.error("Username not found")
        else:
            st.subheader("Sign Up")
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            
            if st.button("Submit Signup"):
                if new_username in user_db:
                    st.error("Username already exists")
                else:
                    user_db[new_username] = pwd_context.hash(new_password)
                    save_user_db(user_db)  # Save user_db to file
                    st.success("Sign Up successful! You can now login.")

    # Prediction Page
    elif selected == "Prediction":
        if "authenticated" in st.session_state and st.session_state["authenticated"]:
            st.title("Traffic Sign Classifier")
            st.write("Choose an option to capture an image for classification:")

            # Option for image upload
            uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                img = Image.open(uploaded_file)
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("Classifying...")

                # Prediction
                class_name = predict_image(img)
                st.write(f"Prediction: {class_name}")

                # Draw bounding box (for visualization)
                detected_boxes = detect_objects(img)
                for box in detected_boxes:
                    img = draw_bounding_box(img, box, class_name)
                st.image(img, caption=f'Image with Bounding Box: {class_name}', use_column_width=True)

            # Option for webcam capture
            webcam_toggle = st.checkbox("Turn on webcam")

            if webcam_toggle:
                st.write("Webcam is active. Capture an image.")
                img = st.camera_input("Capture Image")
                
                if img is not None:
                    img = Image.open(img)
                    st.image(img, caption='Captured Image', use_column_width=True)
                    st.write("Classifying...")
                    
                    # Prediction
                    class_name = predict_image(img)
                    st.write(f"Prediction: {class_name}")

                    # Draw bounding boxes on the image
                    detected_boxes = detect_objects(img)
                    for box in detected_boxes:
                        img = draw_bounding_box(img, box, class_name)
                    st.image(img, caption=f'Captured Image with Bounding Box: {class_name}', use_column_width=True)

            if st.button('Reset'):
                st.experimental_rerun()
        else:
            st.warning("Please log in to access this page.")

    # About Page
    elif selected == "About":
        st.title("About")
        col5, col6 = st.columns([1.5, 1.3])
        col3, col4 = st.columns([2, 3])
        col7, col8 = st.columns([0.9, 0.5])
        col9, col10 = st.columns([0.3, 0.9])

        with col5:
            st.markdown("##### Welcome to Traffic Sign Identifier, your advanced tool for recognizing and interpreting traffic signs to enhance autonomous vehicle navigation. This platform is designed to help vehicles identify various traffic signs in real-time, ensuring safe and efficient driving. Empowering autonomous systems with accurate traffic sign recognition, this tool helps in making informed driving decisions and adapting to road conditions dynamically.")
        with col6:
            st.image("bg.jpeg", use_column_width=True)

        with col3:
            st.image("giphy.webp", use_column_width=True)
        with col4:
            st.header("Key Features")
            st.markdown("""
                        - **Traffic Sign Recognition:** Effortlessly identify traffic signs by uploading images or using a webcam. Get detailed descriptions and information about each sign to understand their meanings and implications for driving.
                        - **Sign Information:** Access a comprehensive database of traffic signs and their regulations. Learn about the purpose, importance, and actions required for various road signs.
                        - **User-Friendly Interface:** The dashboard features a clean, intuitive design, making it easy for users to navigate and interact with the traffic sign recognition system.
                        - **Visual Analytics:** Leverage visual analytics to monitor and analyze the effectiveness of traffic sign recognition. This helps in understanding sign recognition accuracy and improving navigation systems.""")

        with col7:
            st.header("How It Works")
            st.markdown("""**Upload Image:** Users can easily upload an image of a traffic sign using the file upload feature. This allows the system to analyze the sign and provide real-time recognition and classification. Simply drag and drop or select an image file from your device to get started with the traffic sign identification process.""")
            st.markdown("""
                        - **Upload Live Video:** Users can turn on the webcam or mobile camera and upload live video feed of traffic signs.
                        - **View Prediction:** Once an image of a traffic sign is uploaded, the dashboard processes the image and displays the predicted traffic sign along with its corresponding class. This feature provides users with immediate feedback on the traffic sign’s identification, helping to ensure that they can quickly understand the type of sign and its meaning. 
                        - **Learn About the Traffic Sign:** After viewing the prediction, users can access detailed information about the identified traffic sign. This includes its meaning, the actions required in response to the sign, and any relevant traffic rules or regulations.""")
        with col8:
            st.image("bg3.png", use_column_width=True)


if __name__ == "__main__":
    main()
