import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import spacy
import gensim.downloader as api
import lightgbm as lgb
import openai


    
wv = api.load('word2vec-google-news-300')
nlp = spacy.load("en_core_web_lg") 

# Load the pickled files
with open("emotions_model.pkl", "rb") as f:
    emotion_model = pickle.load(f)

openai.api_key='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'
def preprocess_and_vectorize(text):
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_tokens.append(token.lemma_)

    return wv.get_mean_vector(filtered_tokens)





def predict_emotion(text, model):
    """Predict emotion for a given text."""
   
    vector = preprocess_and_vectorize(text)
    vector = np.expand_dims(vector,axis=0)
    prediction = model.predict(vector)
    probabilities = model.predict_proba(vector)
    return prediction[0], probabilities[0]


st.image('Image to be used on dashboard.jpg')
# Streamlit interface
st.title("Russia Ukraine War - Emotion Detection Dashboard")
st.write("This dashboard predicts the emotion behind a text based on models trained on Twitter and Reddit data.")

user_input = st.text_area("Text:")

button_click=False
if st.button("Predict"):
    button_click=True
    

if button_click:
    button_click=False
    with st.spinner('Processing.....'):
        prediction, probs = predict_emotion(user_input,emotion_model)

        classes = emotion_model.classes_ 
        emoji_classes={"anger":" 🤬",
"disgust":"🤢",
"fear":"😨",
"joy":"😀",
"neutral":"😐",
"sadness":"😭",
"surprise":"😲"}
        
        # Getting sorted indices based on probabilities
        sorted_indices = np.argsort(probs)[::-1]
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_classes = [i+emoji_classes[i] for i in sorted_classes]
        sorted_probs = [probs[i] for i in sorted_indices]
        print(prediction)
        response = openai.Completion.create(
            model='text-davinci-003',
            prompt=f'''
            The emotion of a twitter/Reddit user is predicted to be {prediction} with regards to 
            Russia Vs Ukraine conflict based on sentiment analysis, using bullet points give some 
            insights as to why he could be {prediction}
            ''',
            n=1,
            max_tokens=500
        )
    st.write(f"Predicted emotion: **{prediction}**")

    prob_table = {"Emotion Class": sorted_classes, "Probability": [f"{prob:.4f}" for prob in sorted_probs]}
    st.table(prob_table)

    plt.figure(figsize=(10,5))
    plt.bar(sorted_classes, sorted_probs)
    plt.xlabel('Emotion Class')
    plt.ylabel('Probability')
    plt.title('Predicted Probabilities for each Emotion Class')
    st.pyplot(plt)
    st.header('Recommendations')
    st.success(response['choices'][0]['text'])



