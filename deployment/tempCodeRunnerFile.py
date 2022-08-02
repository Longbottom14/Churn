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
    app.run(debug=True)#, host='0.0.0.0', port=9696)