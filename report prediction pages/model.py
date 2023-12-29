from app import db
 
class User(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    name = db.Column(db.String(20),unique=True,nullable=False)
    email = db.Column(db.String(120),unique=True,nullable=False)
    password = db.Column(db.String(60),unique=True,nullable=False)
    def __repr__(self):
        return("Name-(%s) email-{%s}"%(self.name,self.email))
 
class Post(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    disease=db.Column(db.String(20),nullable=False)
    def __repr__(self):
        return("User Disease-(%s)"%(self.disease))
 
8.3 For Website:
1) forms.py
The login and registration form is made using FlaskForm and wtforms. 
 
from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField , SubmitField, BooleanField
from wtforms.validators import DataRequired, Length,Email,EqualTo,ValidationError
class RegistrationForm(FlaskForm):
        	username = StringField("Username", validators=[DataRequired(), Length(min=2,max=20)])
        	email = StringField("Email", validators=[DataRequired() , Email()])
password=PasswordField("Password",validators=[DataRequired()])
confirm_password=PasswordField("Confirm Password",validators=[DataRequired(),EqualTo("password")])
submit= SubmitField("Sign Up ")
class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired() , Email()])
    password=PasswordField("Password",validators=[DataRequired()])
    remember = BooleanField("Remember Me")
    submit= SubmitField(" Login ")
