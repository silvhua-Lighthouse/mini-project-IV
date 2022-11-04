from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
api = Api(app)

# Create a function to create log transformations of Total_Income and LoanAmount
def transform_features(df):
    """
    - Returns a dataframe with:
        - Log of the Total_Income
        - Log of the LoanAmount
        - Transformed features dropped
    """
    # Log of loan amount
    df['LoanAmount_log'] = np.log(df['LoanAmount_log'].astype('float64')) #df['LoanAmount'].apply(np.log)

    # Total income and log of total income
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    transformed_feats = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Total_Income']
    df['Total_Income_log'] = np.log(df['Total_Income_log'].astype('float64')) #df['Total_Income'].apply(np.log)
    df = df.drop(columns=transformed_feats)
    df.columns = df.columns.str.replace('_log', '')
    return df

model = pickle.load(open('/Users/silvh/OneDrive/lighthouse/projects/mini-project-IV/model_random_forest.sav', "rb" ) )

class predict(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        # getting predictions from our model.
        # it is much simpler because we used pipelines during development
        res = model.predict(df)
        # we cannot send numpt array as a result
        return res.tolist() 

# assign endpoint
api.add_resource(predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)