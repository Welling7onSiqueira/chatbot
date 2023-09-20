import json 
import numpy as np
from tensorflow import keras
import pickle

with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)


def chat():
    model = keras.models.load_model('chat_model')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    max_len = 200
    
    while True:
        print("User: ", end="")
        inp = input().lower()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]), truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print("ChatBot:", np.random.choice(i['responses']))

print("Começando interação com o chatbot")
print("Olá, seja bem vindo ao portfólio")
chat()