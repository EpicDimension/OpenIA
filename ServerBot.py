# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:12:39 2022

@author: METALY
"""

import flask
import tiktoken
from flask import request, session
from flask_session import Session
import numpy as np
import openai
import json
import os

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Configurar OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
login = False

# Inicializar el servidor Flask
app = flask.Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

tokenizer = tiktoken.get_encoding("cl100k_base")
gptoken = tiktoken.get_encoding("gpt2")
if login: openai.api_key = ""

@app.route("/login", methods=["POST", "GET"])
def loginpage():
    if request.method == "POST":
        session["api_key"] = request.form.get("api_key")
        return flask.redirect("/")
    return flask.render_template("login.html")

@app.route("/logout", methods=["GET"])
def logout():
    session["api_key"] = ""
    return flask.redirect("/")

@app.route("/gpt2", methods=["POST"], defaults={'Len': False})
@app.route("/gpt2/len", methods=["POST"], defaults={'Len': True})
def gpt2_tokenizer(Len = True):
    data = flask.request.json
    message = data["message"]
    if not message: return ""

    if Len:
        return len(gptoken(message))
    return json.dumps(gptoken(message))

@app.route("/cl100k", methods=["POST"], defaults={'Len': False})
@app.route("/cl100k/len", methods=["POST"], defaults={'Len': True})
def cl100k_tokenizer(Len = True):
    data = flask.request.json
    message = data["message"]
    if not message: return ""
    
    if Len:
        return len(tokenizer(message))
    return json.dumps(tokenizer(message))

@app.route("/gpt3", methods=["POST"])
def gpt3_completion():
    data = flask.request.json
    message = data["message"]
    if not message: return ""

    if not "n" in data: data["n"] = 1
    if not "tokens" in data: data["tokens"] = 512
    if not "temperature" in data: data["temperature"] = 0.7
    if not "presence" in data: data["presence"] = 0.5
    if not "frequency" in data: data["frequency"] = 0.5
    if not "bestof" in data: data["bestof"] = data["n"]
    if not "stop" in data: data["stop"] = False
    
    try: 
        if login:
            openai.api_key = session.get("api_key")
        response = openai.Completion.create(
            prompt= message,
            n=data["n"],
            engine="text-davinci-003",
            temperature=data["temperature"],
            presence_penalty=data["presence"],
            frequency_penalty=data["frequency"],
            best_of=data["bestof"] or data["n"],
            stop=["\n"] if data["stop"] else None,
            max_tokens=data["tokens"] or 512 )
    except: response = {"choices":
            [{"text": "\nError"}]}
    if login: openai.api_key = ""

    # Devolver la respuesta generada por GPT-3 al cliente
    return json.dumps([c["text"] for c in response["choices"]])

@app.route("/")
def home():
    if login and not session.get("api_key"):
        return flask.redirect("/login")
    return flask.render_template("index.html")

# Iniciar el servidor web
if __name__ == "__main__":
    ip = "0.0.0.0" if login else "127.0.0.1"
    app.run(host = ip)
    
