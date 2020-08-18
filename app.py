from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
# This file should read json file and retun all col names in json file
import json
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
__model = None



@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


@app.route("/predicts", methods=['POST'])
def predicts():
    if request.method == 'POST':
        
        jobType = request.form['jobType']
        degree = request.form['degree']
        major = request.form['major']
        industry = request.form['industry']
        yearsExperience = int(request.form['yearsExperience'])
        milesFromMetropolis = int(request.form['milesFromMetropolis'])
        
        pkl_file = open('cat', 'rb')
        index_dict = pickle.load(pkl_file)
        cat_vector = np.zeros(len(index_dict))
        
        try:
            cat_vector[index_dict['jobType_'+str(jobType)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['degree_'+str(degree)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['major_'+str(major)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['industry_'+str(industry)]] = 1
        except:
            pass
        try:
            cat_vector[index_dict['yearsExperience'+int(yearsExperience)]] = yearsExperience
        except:
            pass
        try:
            cat_vector[index_dict['milesFromMetropolis'+int(milesFromMetropolis)]] = milesFromMetropolis
        except:
            pass
        
            
            
        #load artifacts
        print("Loading saved artifacts...START")
       
        global __model
        with open("./artifacts/BestModelSalaryPrediction.pkl", 'rb') as f:
            __model = pickle.load(f)
    
        print("loading saved artifacts...DONE")
        
        #get_estimated_salary
        
        response = __model.predict([cat_vector])
        response = np.around(response[0])
    
        return render_template('index.html',prediction_text="Predicted: {}k".format(response))
    
    else:
        return render_template('index.html')
    

if __name__=="__main__":
    
    app.run(debug=True)
    

