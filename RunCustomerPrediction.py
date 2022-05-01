import argparse
import sys
import os
import pandas as pd
import numpy as np
import pickle
import joblib

#Arguments
parser = argparse.ArgumentParser(description="This is Customer segmentation Model")
parser.add_argument("-input_data", '--input_data', action="store", dest='input_data', help="Input Data set required", type=str)
parser.add_argument("-model_path",  '--model_path', action="store", dest='model_path', help="Model path required", type=str)
parser.add_argument("-Output_data", '--Output_data', action="store", dest='Output_data', help="Output Data set required", type=str)
args = parser.parse_args()

#prepare and write output
def transform_data(df, precition, output_path):
    converts  = { 1: 'Will Buy', 0: 'Will Not Buy'}
    predicted_Values = pd.DataFrame(data=precition,columns=['Prediction'])
    predicted_Values['Prediction'] = predicted_Values['Prediction'].map(converts)
    results = pd.concat([df['ID'], predicted_Values],axis=1, ignore_index=True,sort=False)
    results.columns = ['ID','Prediction']
    return results

def predict(inputpath, model_path):
    #load the model pickle file
    loaded_model = joblib.load(model_path)
    #drop ID column from the features
    inputdata = pd.read_csv(inputpath)
    Customer_info = inputdata.drop(['ID'], axis=1).to_numpy()
    y_pred = loaded_model.predict(Customer_info)
    #print(y_pred)
    return y_pred, inputdata

def main():
    predict(args.input_data, args.model_path)
    predictions, features = predict(args.input_data, args.model_path)
    pred = transform_data(features, predictions, args.Output_data)
    pred.to_csv(args.Output_data)
    #print(pred.head(20))
if __name__ == '__main__':
        print("script started")
        main()

