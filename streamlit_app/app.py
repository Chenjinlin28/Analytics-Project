import streamlit as st
from setfit import SetFitModel
import torch

# Load the model
model_path = "/Users/jinlinchen/Documents/Study/HWR Berlin/Semester 2/Analytics Lab/Analytics Project/Database part/model/model2/model"
model = SetFitModel.from_pretrained(model_path)

# Convert class id to class name so that we get a clearer output
class_names = {
    1: "belongs_to_article",
    2: "related_work",
    3: "background_information"
}

# Entity class prediction function
def predict_entity_class(entity, sentence):
    input_data = f"{entity}, {sentence}"
    prediction = model.predict([input_data])
    predicted_class_id = prediction.item() 
    return class_names[predicted_class_id]

# Streamlit app
st.title("Entity Classification App")

st.write("Enter the Entity and the Sentence it is located in")

entity = st.text_input("Entity:")
sentence = st.text_area("Sentence:")

if st.button("Predict"):
    if entity and sentence:
        prediction = predict_entity_class(entity, sentence)
        st.write(f"Predicted Entity Class: {prediction}")
    else:
        st.write("Please enter both the entity and the sentence.")