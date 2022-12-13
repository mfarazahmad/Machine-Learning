import joblib
import numpy as np

#Load the model from the file
filename = './ml_model_example_regression.pkl'
loaded_model = joblib.load(filename)

#Retreive New Data to Predict With (Psuedo Code)
# az = Data.Connect())
# x_new = az.blob()
x_new = np.array([0,1,1,0,0,1,0.344167,0.363625,0.805833,0.160446]).reshape(-1, 1)

result = loaded_model.predict(x_new)
print('Prediction: {:.0f} '.format(np.round(result[0])))