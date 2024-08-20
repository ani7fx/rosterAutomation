from flask import Flask, request, app, jsonify, render_template, url_for
import numpy as np
import pandas as py

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/generate_roster_api', methods=['POST'])
def generate_roster_api():
    data = request.json['data']
    print(data)
    