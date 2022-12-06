import joblib
import pandas as pd
from flask import Flask
from flask import jsonify
from flask import request
from flask import Response
from flask_cors import CORS, cross_origin
from sklearn.preprocessing import LabelBinarizer
import json
import config


def create_app(arg_environment):
    local_app = Flask(__name__)
    local_app.config.from_object(arg_environment)
    return local_app


environment = config
app = create_app(environment)
CORS(app, support_credentials=True)


@app.route('/ping', methods=['GET'])
def get_ping():
    return 'pong'


@cross_origin(supports_credentials=True)
@app.route('/predict', methods=['POST'])
def post_predict():
    version = request.args.get('model')
    print(version)
    requestBody = request.json
    dfToPredict = pd.DataFrame.from_records(requestBody)
    model: any
    if version is None or version != 'V2':
        model = joblib.load('models/best_model.joblib')
    else :
        model = joblib.load('models/second_model.joblib')
    results = model.predict(dfToPredict)
    i = 0
    customerList = []
    for customer in requestBody:
        cust = {
            'customerID' : customer['customerID'],
            'churn': int(results[i])
        }
        i += 1
        customerList.append(cust.copy())

    return Response(json.dumps(customerList),
                    status=200,
                    mimetype="application/json")



@cross_origin(supports_credentials=True)
@app.route('/train', methods=['POST'])
def post_train():
    requestBody = request.json
    dfToTrain = pd.DataFrame.from_records(requestBody)
    dfToTrain["TotalCharges"] = pd.to_numeric(dfToTrain["TotalCharges"], errors='coerce')
    dfToTrain['TotalCharges'].fillna(0, inplace=True)
    dfToTrain.Churn = LabelBinarizer().fit_transform(dfToTrain.Churn)
    model = joblib.load('models/best_model.joblib')
    
    features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

    X = dfToTrain[features]
    Y = dfToTrain['Churn']

    model.fit(X,Y)

    score = model.score(X, Y)
    joblib.dump(model, 'models/second_model.joblib')

    return jsonify([{'First Model ROC AUC': 0.735435 },{'Second Model ROC AUC': score}])



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
