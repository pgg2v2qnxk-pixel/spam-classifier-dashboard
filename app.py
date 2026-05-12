
import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("Spam Message Classification Dashboard")

st.write("Enter a message below to classify it as Spam or Ham.")

# Input box
message = st.text_area("Enter Message")

# Predict button
if st.button("Check Message"):

    transformed_message = vectorizer.transform([message])

    prediction = model.predict(transformed_message)

    if prediction[0] == 1:
        st.error("🚫 Spam Message")
    else:
        st.success("✅ Legitimate Message")
