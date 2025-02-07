import os
os.environ["OMP_NUM_THREADS"] = "1"  # Set environment variable to avoid memory leak

from flask import Flask, jsonify
import pandas as pd
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the data
data = pd.read_csv('D:/WebiSoftTech/K-MEANS ML/Income/income.csv')

# Prepare the data for clustering
income_data = data[['Income($)']]

# Fit K-Means model with explicit n_init
kmeans = KMeans(n_clusters=4, n_init=10)  # Set n_init explicitly
data['Income Group'] = kmeans.fit_predict(income_data)

@app.route('/income_groups', methods=['GET'])
def get_income_groups():
    # Return the data with income groups
    return jsonify(data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)