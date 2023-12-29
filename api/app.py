import joblib
import numpy as np
from sklearn.metrics import accuracy_score
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
 
from chat import get_response
app=Flask(__name__,template_folder='template') 
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
from keras.models import load_model
model = load_model(r"C:\Users\Dell\project\model111.h5")
model222 = load_model(r"C:\Users\Dell\project\my_model.h5")
def api(full_path):
    data = image.load_img(full_path, target_size=(50, 50, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model.predict(data)
    return predicted
def api1(full_path):
    data = image.load_img(full_path, target_size=(64, 64, 3))
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255
    predicted = model222.predict(data)
    return predicted
#MALARIA MODEL
@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            result = api(full_name)
            print(result)
 
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")
            return redirect(url_for("Malaria"))

#PNEUMONIA MODEL
@app.route('/upload11', methods=['POST','GET'])
def upload11_file():
    if request.method == 'GET':
        return render_template('index2.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)
            indices = {0: 'Pneumonia', 1: 'Normal'}
            result = api1(full_name)
            print(result)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('predict1.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except:
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("Pneumonia"))
@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
#LOGIN AND REGISTRATION
from forms import RegistrationForm, LoginForm
@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        flash("Account created","success")      
        return redirect(url_for("home"))
    return render_template("register.html", title ="Register",form=form )
@app.route("/login", methods=["POST","GET"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        if form.email.data =="sho" and form.password.data=="password":
            flash("You Have Logged in !","success")
        return redirect(url_for("home"))
    return render_template("login.html", title ="Login",form=form )
def ValuePredictor1(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,30)
    loaded_model = joblib.load(r"C:\Users\Dell\project\model")
    result = loaded_model.predict(to_predict)
    return result[0]
#PAGES
@app.route("/")
@app.route("/mainpage")
def mainpage():
    return render_template("mainpage.html")  
@app.route("/home")
def home():
    return render_template("home.html")
@app.post("/predict_chat")
def predict_chat():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/cancer")
def cancer():
    return render_template("cancer.html")
@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")
@app.route("/heart")
def heart():
    return render_template("heart.html")
@app.route("/liver")
def liver():
    return render_template("liver.html")
@app.route("/kidney")
def kidney():
    return render_template("kidney.html")
@app.route("/Malaria")
def Malaria():
    return render_template("index.html")
@app.route("/Pneumonia")
def Pneumonia():
    return render_template("index2.html")
 
#PREDICTION
def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==8):#Diabetes
        loaded_model = joblib.load(r"C:\Users\Dell\project\model1")
        result = loaded_model.predict(to_predict)
    elif(size==30):#Cancer
        loaded_model = joblib.load(r"C:\Users\Dell\project\model")
        result = loaded_model.predict(to_predict)
    elif(size==12):#Kidney
        loaded_model = joblib.load(r"C:\Users\Dell\project\model3")
        result = loaded_model.predict(to_predict)
    elif(size==10):#Liver
        loaded_model = joblib.load(r"C:\Users\Dell\project\model4")
        result = loaded_model.predict(to_predict)
    elif(size==11):#Heart
        loaded_model = joblib.load(r"C:\Users\Dell\project\model2")
        result =loaded_model.predict(to_predict)
    return result[0]
@app.route('/result',methods = ["POST"])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if(len(to_predict_list)==30):#Cancer
            result = ValuePredictor(to_predict_list,30)
        elif(len(to_predict_list)==8):#Diabetes
            result = ValuePredictor(to_predict_list,8)
        elif(len(to_predict_list)==12):#Heart
            result = ValuePredictor(to_predict_list,12)
        elif(len(to_predict_list)==11):#Kidney
            result = ValuePredictor(to_predict_list,11)
        elif(len(to_predict_list)==10):#Liver
            result = ValuePredictor(to_predict_list,10)
    if(int(result)==1):
        prediction = "Sorry! The result is Positive"
        info = "We advise you to seek medical expertise as soon as you can. We hope you get well soon!"
    else:
        prediction = "Congrats! The result is Negative!"
        info = "However, precaution is better than cure. So, stay healthy and exercise daily!"
    return(render_template("result.html", prediction=prediction, info=info))
 
#CHATBOT PAGE
@app.route('/chatbot')
def index():
    data = []
    user_symptoms.clear()
    file = open("static/ds_symptoms.txt", "r")
    all_symptoms = file.readlines()
    for s in all_symptoms:
        data.append(s.replace("'", "").replace("_", " ").replace(",\n", ""))
    data = json.dumps(data)
    return render_template('style_chat.html', data=data)
#FLASK RUN
if __name__ == "__main__":
    app.debug = True
    app.run()
 
#CHATBOT IMPLEMENTATION
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
            description = diseases_description.loc[diseases_description['Disease'] == disease.strip(" ").lower(), 'Description'].iloc[0]
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


