import json
import torch
import nltk
import pickle
import random
from datetime import datetime
import numpy as np
import pandas as pd
from nnet import NeuralNet
from nltk_utils import bag_of_words
from flask import Flask, render_template, url_for, request, jsonify
random.seed(datetime.now())
device = torch.device('cpu')
FILE = "data_chat.pth"
model_data = torch.load(FILE)
 
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data['model_state']
 
nlp_model = NeuralNet(input_size, hidden_size, output_size).to(device)
nlp_model.load_state_dict(model_state)
nlp_model.eval()
diseases_description = pd.read_csv("data/symptom_Description.csv")
diseases_description['Disease'] = diseases_description['Disease'].apply(lambda x: x.lower().strip(" "))
disease_precaution = pd.read_csv("data/symptom_precaution.csv")
disease_precaution['Disease'] = disease_precaution['Disease'].apply(lambda x: x.lower().strip(" "))
symptom_severity = pd.read_csv("data/Symptom-severity.csv")
symptom_severity = symptom_severity.applymap(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)
docprof = pd.read_csv("data/doctors_dataset.csv")
docprof = docprof.applymap(lambda s: s.lower().strip(" ").replace(" ", "") if type(s) == str else s)
with open('data/list_of_symptoms.pickle', 'rb') as data_file:
    symptoms_list = pickle.load(data_file)
with open('fitted_model.pickle', 'rb') as modelFile:
    prediction_model = pickle.load(modelFile)
user_symptoms = set()
 
def get_symptom(sentence):
    sentence = nltk.word_tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = nlp_model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    prob = prob.item()
    return tag, prob
 
@app.route('/symptom', methods=['GET', 'POST'])
def predict_symptom():
    print("Request json:", request.json)
    sentence = request.json['sentence']
    if sentence.replace(".", "").replace("!","").lower().strip() == "done":
 
        if not user_symptoms:
            response_sentence = random.choice(
                ["I can't know what disease you may have if you don't enter any symptoms.",
                "Meddy can't know the disease if there are no symptoms...",
                "You first have to enter some symptoms!"])
        else:
            x_test = []
            for each in symptoms_list:
                if each in user_symptoms:
                    x_test.append(1)
                else:
                    x_test.append(0)
            x_test = np.asarray(x_test)            
            disease = prediction_model.predict(x_test.reshape(1,-1))[0]
            print(disease)
        	description=diseases_description.loc[diseases_description['Disease'] == disease.strip(" ").lower(), 'Description'].iloc[0]
        	precaution = disease_precaution[disease_precaution['Disease'] == disease.strip(" ").lower()]
        	precautions = 'Precautions: ' + precaution.Precaution_1.iloc[0] + ", " + precaution.Precaution_2.iloc[0] + ", " + precaution.Precaution_3.iloc[0] + ", " + precaution.Precaution_4.iloc[0]
            symptom, prob = get_symptom(sentence)
            confidence_level = (1.0*len(symptom))/len(symptoms_list)
            severity = []
            for each in user_symptoms:
                severity.append(symptom_severity.loc[symptom_severity['Symptom'] == each.lower().strip(" ").replace(" ", ""), 'weight'].iloc[0])
            response_sentence = f"It looks to me like you have <b>" + disease + "</b>. <br><br> <i>Description: " + description + "</i>" + "<br><br><b>"+ precautions + "</b><br><br>" + f"Confidence level: {(confidence_level * 1000):.2f}% "
 
            if np.mean(severity) > 4 or np.max(severity) > 5:
                response_sentence = response_sentence + "<br><br>Considering your symptoms are severe", print(severity) , " you should consider talking to a real doctor!"
            user_symptoms.clear()
            severity.clear()
    else:
        symptom, prob = get_symptom(sentence)
        print("Symptom:", symptom, ", prob:", prob)
        if prob > .5:
            response_sentence = random.choice(
                ["Please continue.",
                "Okay, any more symptoms?",
                "Are there any more symptoms you might be facing?",
                "Hmm, we see. Any more symptoms you could think of?"])
            user_symptoms.add(symptom) 
        elif sentence != symptom:
            if sentence == 'Hi' or 'Hello' or 'hi' or 'hello':
                response_sentence = random.choice(["Hello, Welcome to HealthCare Chatbot. Please state your symptoms.",
                 "Hi, Welcome to HealthCare Chatbot. You can start stating your symptoms",
                    "Hello, nice to see you here. Please state your symptoms.",
                    "Hello. Please let us know what symptoms you are experiencing."])
            if sentence != 'Hi' or 'Hello' or 'hi' or 'hello':
                if sentence == 'thank you':
                    response_sentence = "Your welcome. We hope you get well soon."
                if sentence == 'thanks':
                    response_sentence = "Your welcome. We hope you get well soon."
                if sentence == 'okay':
                    response_sentence = "Happy to help. We hope you get well soon."
                if sentence == 'Okay':
                    response_sentence = "Happy to help. We hope you get well soon."
        else:
            response_sentence = "I'm sorry, but I don't understand you."
        print("User symptoms:", user_symptoms)
    return jsonify(response_sentence)
