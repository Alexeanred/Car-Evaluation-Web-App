import time
import pandas as pd
import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

st.write("""# CAR EVALUATION APP
For more details, please visit this page: [Car Evaluation Dataset](http://archive.ics.uci.edu/ml/datasets/Car+Evaluation)
or [Kaggle Car Evaluation Dataset](https://www.kaggle.com/mykeysid10/car-evaluation/data)""")

st.image('Car.jpg',width=600)

#sidebar part
st.sidebar.header("User Input Features")
def user_input_features():
    buying = st.sidebar.radio("Buying price",['low', 'med', 'high', 'vhigh'])
    maintance = st.sidebar.radio("Price of the maintance",['low', 'med', 'high', 'vhigh'])
    doors = st.sidebar.radio("Number of doors",['2', '3', '4', '5more'])
    persons = st.sidebar.radio("Number of persons",['2', '4', 'more'])
    lug_boot = st.sidebar.radio("The size of luggage boot",['small', 'med', 'big'])
    safety = st.sidebar.radio("Safety of the car",['low', 'med', 'high'])
    data = {'buying': buying,
            'maintance': maintance,
            'doors': doors,
            'persons': persons,
            'lug_boot': lug_boot,
            'safety': safety}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()
    
st.subheader("User Input Features")
st.write(input_df)

car_evaluation_raw = pd.read_csv('car_evaluation.csv')
car_df = car_evaluation_raw.drop(columns=['class'])

categories = [car_df[i].unique() for i in car_df.columns]
categories[0] = categories[0][::-1]
categories[1] = categories[1][::-1]
categories[-1][2:] = categories[-1][2:][::-1]

oe = OrdinalEncoder(categories= categories)
input_df[car_df.columns] = oe.fit_transform(input_df[car_df.columns])

#load the builded classification model
load_clf = pickle.load(open('car_evaluation_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

def imagePrediction(x):
    if x=="unacceptable":
        st.image('unacc.jpg',width=600)
    elif x=="acceptable":
        st.image('acc.jpg',width=600)
    elif x=="good":
        st.image('good.jpg',width=600)  
    elif x=="very good":
        st.image('verygood.jpg',width=600)  
#display the predicted result
Class = np.array(["unacceptable","acceptable","good","very good"])
if st.sidebar.button("Finish input"):
#prediction part     
    st.subheader('Prediction')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.00000000001)
        my_bar.progress(percent_complete + 1)
    st.write(f"""The evaluation based on the above data: This is a(n) **{Class[int(prediction)]}** car""")
    imagePrediction(Class[int(prediction)])
#prob part
    st.subheader('Prediction Probability')
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.00000000000000000000001)
        my_bar.progress(percent_complete + 1)
    x = np.array([f"{round(prediction_proba[0][i]*100,1)} %" for i in range(len(prediction_proba[0]))]).reshape(1,-1)
    df = pd.DataFrame(x,columns=["unacceptable","acceptable","good","very good"])
    st.dataframe(df) 



