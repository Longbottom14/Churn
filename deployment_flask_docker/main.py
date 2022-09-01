import pandas as pd
import numpy as np
import pickle
import ast #to convert string to dictionary
from flask import Flask, request, jsonify,render_template,redirect ,url_for

import warnings 
warnings.filterwarnings('ignore')

loaded_models ={} # load models
with open('churn-models.bin', 'rb') as f_in:
    loaded_models['xgb'],loaded_models['Lgb'],loaded_models['Logistic_reg'],loaded_models['bayes'] = pickle.load(f_in)

def preprocessing_single(single_dict):
    df = pd.DataFrame(single_dict,index=[0])

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    #df.churn = (df.churn == 'yes').astype(int)
    
    df['totalcharges'] = pd.to_numeric(df['totalcharges'], errors='coerce')
    df['totalcharges'] = df['totalcharges'].fillna(0)
    return df

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
               'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection',
               'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']

def predict_(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='records')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

def predict_single(trained_models,df_single):
    preds_table_single = pd.DataFrame()
    for model_name in trained_models: 
            #print(f"==========={model_name}==========")
            model = trained_models[model_name]['model_']
            dv = trained_models[model_name]['dv_']
            
            y_pred_single = predict_(df_single, dv, model)
            
            preds_table_single[model_name] = y_pred_single
            #preds_single =  (y_pred_single>= 0.5)*1
    p_df = preds_table_single.copy()
    p_df['blend3'] = 0.4* p_df.Logistic_reg + 0.4*p_df.bayes + 0.1*p_df.xgb + 0.1*p_df.Lgb
    #p_df['blend10'] = 0.3* p_df.Logistic_reg + 0.5*p_df.bayes + 0.1*p_df.xgb + 0.1*p_df.Lgb
    preds_single =  (p_df['blend3']>= 0.5)*1
    
    return     p_df['blend3'].values[0] #, preds_single[0]


app = Flask('churn')
@app.route('/')
def welcome():
    return render_template('index.html') #homepage

@app.route('/submit',methods=['POST'])
def submit():
    #return " submit page"
    if request.method == 'POST':
        #gen = str(request.form['gender'])
        #fetching data from the form
        customer = {'customerID': '5575-GNVDE',
                        'gender': str(request.form['gender']),
                        'SeniorCitizen': str(request.form['seniorcitizen']),
                        'Partner': str(request.form['partner']),
                        'Dependents': str(request.form["dependents"]),
                        'tenure': float(request.form['tenure']),
                        'PhoneService': str(request.form["phoneservice"]),
                        'MultipleLines': str(request.form["multiplelines"]),
                        'InternetService': str(request.form["internetservice"]),
                        'OnlineSecurity': str(request.form["onlinesecurity"]),
                        'OnlineBackup': str(request.form["onlinebackup"]),
                        'DeviceProtection': str(request.form["deviceprotection"]),
                        'TechSupport': str(request.form['techsupport']),
                        'StreamingTV': str(request.form['streamingtv']),
                        'StreamingMovies': str(request.form['streamingmovies']),
                        'Contract': str(request.form['contract']),
                        'PaperlessBilling': str(request.form['paperlessbilling']),
                        'PaymentMethod': str(request.form['paymentmethod']),
                        'MonthlyCharges':float(request.form['monthlycharges']),
                        'TotalCharges': str(request.form['totalcharges']),
                        #'Churn': 'No'
                        
                     }
    #return jsonify(customer)
    customer = str(customer) # convert dictionary to string

    return redirect(url_for('predict' , customer = customer)) # go to predict page and dump the str_dict

@app.route('/predict/<string:customer>') # <string: str_dict>
def predict(customer):
    customer = ast.literal_eval(customer) # str to dict
    clean_customer = preprocessing_single(single_dict=customer)
    
    prediction = predict_single(trained_models=loaded_models,df_single=clean_customer)

    #prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5
    
    result = {
        'churn_probability': float(np.round(prediction,3)),
        'churn': bool(churn)
    }
    topk = ['TotalCharges','tenure','Contract','PaymentMethod']
    return render_template('result.html',customer_passer = customer , best_predictors_passer = topk , result_passer = result)#return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
