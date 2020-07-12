from flask import Flask,request,jsonify,render_template,request,Response,redirect,flash, url_for
from config import Config
import pandas as pd 
from flask_cors import CORS
from modules import urlparser,makepredictions,performbuild
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)
@app.route("/api", methods=['POST','GET'])
def api():
    return jsonify({'message' : 'Spam url Checking api Working'}) , 200

@app.route("/api-link")
def link():
    return render_template("api.html")
@app.route('/api/predict',methods=['POST'])
def predicter():
    if request.method == 'POST':
        message = request.json['message']
        urls = urlparser(message)
        result = makepredictions(urls)
        return jsonify(result),200
@app.route("/", methods=['POST','GET'])
def home():
    if request.method == 'POST':
        message = request.form['message']
        urls = urlparser(message)
        result = makepredictions(urls)
        if result:
            return render_template("home.html",result=result)
        else:
            flash("Sorry no urls found.","danger")
            return render_template("home.html",result=False)
    else:
        return render_template("home.html",result=False)
@app.route("/build", methods=['POST','GET'])
def builder():
    score = performbuild()
    return jsonify({'score' : score }) , 200
@app.route("/dataset", methods=['POST','GET'])
def dataset():
    data = pd.read_csv('data.csv',encoding = "ISO-8859-1")
    dataset = list(data.values)
    return render_template('dataset.html', dataset=dataset)
if __name__ == '__main__':
	app.run(debug=True)
