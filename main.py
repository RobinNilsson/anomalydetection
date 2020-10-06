import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#1

df = pd.read_csv('Data/signal.csv', encoding='utf8', engine='python', sep=',', names=['timestamp', 'value', 'tail'])

df['value'] = df['value'].replace([0], np.NaN).interpolate()

df['timestamp'] = pd.to_datetime(df['timestamp'])

signalData = df['value']

#ax = plt.plot(df['value'])
#plt.plot((df['tail']*5)+60, color='red')

dataLength = len(df)

def create_noise_vector(dataLength):
    # Set parameters
    # for anomaly generation

    anomalyLength = 2000
    anomalyAmplitude = 2
    anomalyWavelength = 200 #Data points per period
    anomalyNumber = 50

    # Create single anomaly vector
    singleAnomalyVec = anomalyAmplitude * np.sin(np.array(range(1, anomalyLength))*np.pi / anomalyLength)
    singleAnomalyVec = singleAnomalyVec * 2 * np.sin(np.array(range(1, anomalyLength))*np.pi / anomalyWavelength)

    plt.plot(singleAnomalyVec)

    # Randomly place a number of single vectors onto start dataset

    anomalyStartVec = np.sort(np.floor(np.random.rand(anomalyNumber) * (dataLength - anomalyLength))).astype(int)

    anomalyVec = np.zeros(dataLength)
    anomalyLabel = np.zeros(dataLength)

    for ii in range(anomalyNumber):
        anomalyVec[anomalyStartVec[ii]:anomalyStartVec[ii]+anomalyLength-1] = singleAnomalyVec
        anomalyLabel[anomalyStartVec[ii]:anomalyStartVec[ii]+anomalyLength-1] = 1

    plt.plot(anomalyVec)

    return anomalyVec, anomalyLabel


# Get anomaly vector
anomalyVec, anomalyLabel = create_noise_vector(dataLength)

# Superpose anomaly vector
signalDataAnomaly = signalData + anomalyVec

# Add new outliers
outlierVec = np.random.rand(dataLength)

signalDataAnomaly[outlierVec < 0.0001] = 0

ax = plt.plot(signalDataAnomaly)

plt.plot(anomalyLabel*5+40, color='red')

# Save data
generated_data = pd.DataFrame()
generated_data['timestamp'] = df['timestamp']
generated_data['signalDataAnomaly'] = pd.Series(signalDataAnomaly)
generated_data['anomalyLabel'] = pd.Series(anomalyLabel)

generated_data.to_pickle('generated_data')
