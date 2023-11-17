### import neccesary library
import streamlit as st  ### used for visualization
import numpy as np  
import random ### for generating random data
from statsmodels.tsa.seasonal import STL  ### used STL method to detect anomoly
import pandas as pd
import matplotlib.pyplot as plt



def generate_data_stream(data_type='Linear', length=1000, seasonality_period=50, anomaly_chance=0.01):
    ## add linear, sinewave, randowwalk data stream
    if data_type == 'Linear':
        base_pattern = np.linspace(0, 10, length)
    elif data_type == 'Sine Wave':
        base_pattern = np.sin(np.linspace(0, 2 * np.pi, length))
    elif data_type == 'Random Walk':
        base_pattern = np.cumsum(np.random.normal(loc=0.5, size=length))
    else:
        st.error("Invalid data type")
        return
    #### This function is used to genrate a stream data, as here i tried to different-2 type of where anomoly can be present

    seasonal_pattern = np.sin(np.linspace(0, 2 * np.pi, seasonality_period))
    ### add seasonility pattern
    seasonal_component = np.tile(seasonal_pattern, length // seasonality_period + 1)[:length]
    noise = np.random.normal(0, 0.5, length)
    ### add noise in data stream
    data_stream = base_pattern + seasonal_component + noise

    for i in range(length):
        if random.random() < anomaly_chance:
            data_stream[i] += np.random.normal(15, 5)

    return data_stream
### function to add noise in stream
def add_noise(data_stream, noise_level=1.0):
    return data_stream + np.random.normal(0, noise_level, len(data_stream))
### function to add seasonality in steam
def add_seasonality(data_stream, seasonality_period=50, amplitude=5.0):
    seasonal_pattern = np.sin(np.linspace(0, 2 * np.pi, seasonality_period))
    seasonal_component = np.tile(seasonal_pattern, len(data_stream) // seasonality_period + 1)[:len(data_stream)]
    return data_stream + amplitude * seasonal_component
### function to detect anomalies using STL method z score - 3 thresold (99.7%)
def detect_anomalies(data_stream, seasonality_period=50, z_score_threshold=3.0):
    ### data point will be anomalies if z-value is greater than thresold
    stl = STL(pd.Series(data_stream), period=seasonality_period)
    result = stl.fit()
    residuals = result.resid
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    anomalies = np.where(z_scores > z_score_threshold)[0]
    return anomalies

### for visualization(using streamlit)
def plot_data_with_anomalies(data_stream, anomalies):
    plt.figure(figsize=(10, 6))
    plt.plot(data_stream, label='Data Stream', color='blue')

    if len(anomalies) > 0:
        plt.scatter(anomalies, data_stream[anomalies], color='red', label='Anomalies')

    plt.title('Data Stream with Anomalies')
    plt.xlabel('Data Point')
    plt.ylabel('Value')
    plt.legend()
    st.pyplot(plt)

def main():
    ### using a streamlit library for visaulization
    st.title('Anomaly Detection in Data Stream')
    anomalies = []
    data_types = ['Linear', 'Sine Wave', 'Random Walk']
    selected_data_type = st.sidebar.selectbox('Select Data Type', data_types)
    data_stream = generate_data_stream(selected_data_type)
    st.line_chart(data_stream, use_container_width=True)
    noise_level = st.sidebar.slider('Select Noise Level', min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    if st.sidebar.button('Add Noise'):
        data_stream = add_noise(data_stream, noise_level)

    seasonality_amplitude = st.sidebar.slider('Select Seasonality Amplitude', min_value=1.0, max_value=10.0, value=5.0, step=0.1)
    if st.sidebar.button('Add Seasonality'):
        data_stream = add_seasonality(data_stream, amplitude=seasonality_amplitude)
    if st.sidebar.button('Find Anomalies'):
        anomalies = detect_anomalies(data_stream)
    plot_data_with_anomalies(data_stream, anomalies)

if __name__ == '__main__':
    main()
