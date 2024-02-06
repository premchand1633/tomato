import streamlit as st
from PIL import Image
import numpy as np
import pickle

st.title("Is this tomato ripe?")

# Load the trained logistic regression model
with open('ripe_or_not_log2.pkl', 'rb') as file:
    model = pickle.load(file)

st.write("This is a deep learning classification app to determine if a tomato is ripe or not.")


ripeTomato = Image.open("ripeTomato.jpg")
ripeTomato = ripeTomato.resize((300, 200))

unripeTomato = Image.open("unripeTomato.jpg")
unripeTomato = unripeTomato.resize((300, 200))



st.write("Here are some examples of ripe and unripe tomatoes:")

st.write(" ")
# show and align images horizontally
col1, col2 = st.columns(2)

with col1:
    st.write("Ripe tomatoes")
    st.image(image=ripeTomato, caption="This is a ripe tomato", width=300)

with col2:
    st.write("Unripe tomatoes")
    st.image(image= unripeTomato, caption="This is an unripe tomato", width=300)


st.write(" ")

# Function to load and preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((192, 192))  # Resize image to match training data
    img_array = np.array(img).reshape(1, -1)  # Flatten image into a 1D array
    return img_array


# Display upload file section
st.write("Upload an image of a tomato to determine if it is ripe or not.")
input_image = st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])

# If an image is uploaded, make predictions
if input_image is not None:
    tomato_image = preprocess_image(input_image)
    prediction = model.predict(tomato_image)
    probability = model.predict_proba(tomato_image)[0][1]  # Probability of being ripe (class 1)

    # Display prediction
    st.image(input_image, caption="This is the tomato you uploaded", use_column_width=True)
    if prediction == 1:
        st.write("The tomato is ripe.")
    else:
        st.write("The tomato is not ripe.")
    st.write(f"Probability it's a ripe tomato: {probability:.4f}")
