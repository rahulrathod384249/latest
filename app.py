from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
app = Flask(__name__,template_folder='template')
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == "POST":
        
        gold_data = pd.read_csv('gld_price_data.csv')
        X = gold_data.drop(['Date','GLD'],axis=1)
        Y = gold_data['GLD']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)
        regressor = RandomForestRegressor(n_estimators=100)
        model = regressor.fit(X_train,Y_train)
        # Get values through input bars
        SPX = request.form.get("SPX")
        USO = request.form.get("USO")
        SLV = request.form.get("SLV")
        EUR = float(request.form.get("EUR"))
        # Put inputs to dataframe
        X = [SPX,USO,SLV,EUR]
        arr = np.array(X).reshape(1,-1)
        # Get prediction
        pred = model.predict(arr)
        prediction = np.round(pred, 4)
    else:
        prediction = ""
    return render_template("index.html", output = prediction)
if __name__ == '__main__':
    app.run()