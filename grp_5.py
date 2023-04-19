import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from pickle import load
from pickle import dump
import warnings
warnings.filterwarnings('ignore')
import nsepy as nse
from datetime import date
import datetime
current_time=datetime.datetime.now()


# Background
# Set page config
st.set_page_config(page_title="My Streamlit App", page_icon=":smiley:", layout="wide")

# Set background image
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('white.webp')
    
# Add title with CSS styles
st.markdown("""
<style>
h1 {
    text-align: center;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

st.title("GROUP 5 Project")


# load the pre-trained LSTM model
loaded_model=pickle.load(open(r'bp_trained_model.sav','rb'))

bp=nse.get_history(symbol='BERGEPAINT',start=date(2010,1,1),end=date(current_time.year,current_time.month,current_time.day))

bp=bp[['Open','High','Low','Close']]

#describe data
st.subheader("Data from 2010 till today(describe data)")
st.write(bp.describe())

#visualization
st.subheader("Closing Price VS Time Chart")
fig = plt.figure(figsize = (12,6))
plt.plot(bp.Close)
st.pyplot(fig)

st.subheader("Closing Price VS Time Chart with 100MA")
ma100 =bp.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(bp.Close)
st.pyplot(fig)

st.subheader("Closing Price VS Time Chart with 100MA & MA200")
ma100 =bp.Close.rolling(100).mean()
ma200 =bp.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(bp.Close,'b')
st.pyplot(fig)




scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bp)

def bp_stockprice(n_future):

    #n_future = input()  # Number of days to forecast
    n_past = 60
    last_30_days = scaled_data[-n_past:]
    X_test = np.array([last_30_days])
    predictions = []

    for i in range(n_future):
        next_day_pred = loaded_model.predict(X_test)[0, 0]
        last_30_days = np.append(last_30_days[1:, :], [[next_day_pred, next_day_pred, next_day_pred, next_day_pred]], axis=0)
        X_test = np.array([last_30_days])
        pred_value = scaler.inverse_transform([[0, 0, 0, next_day_pred]])[0, 3]
        predictions.append(pred_value)
        print("Day {}: {}".format(i+1, pred_value))
    return np.round(predictions,0)




def main():
    #giving title
    st.title('Forcasting future data web app')

    #getting input variable from users
    n_future=st.text_input('Number of future data')

    diagnosis=''

    #creating button for prediction
    if st.button('Future days data predicted'):
        diagnosis=bp_stockprice(int(n_future))
    
    st.success(diagnosis)
if __name__ == '__main__':
    main()



# for grAPH  SPLIT DATA AND ... ..

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(bp)

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data = scaled_data[0:train_size,:]
test_data = scaled_data[train_size:len(scaled_data),:]

def create_dataset(data, time_step=1):
    X_data, y_data = [], []
    for i in range(len(data)-time_step):
        X_data.append(data[i:(i+time_step), :])
        y_data.append(data[i+time_step, :])
    return np.array(X_data), np.array(y_data)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


predictions =loaded_model.predict(X_test)

# predictions = scaler.inverse_transform(loaded_model.predictions)
y_test = scaler.inverse_transform(y_test)





# Final Graph
st.subheader("predictions vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(predictions, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()