import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load signal
df = pd.read_pickle('generated_data')
dataLength = len(df)
anomalyLength = 2000
anomalyWavelength = 200

# variable "signalDataAnomaly" is the signal to be analysed
# Clean from outliers

df['signalDataAnomaly'].loc[df['signalDataAnomaly'] < 50] = np.NaN
df['signalDataAnomaly'] = df['signalDataAnomaly'].interpolate()

signalDataAnomaly = df['signalDataAnomaly']

# Convert datetime to unix time in [s]
time = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

anomalyLabel = df['anomalyLabel']

# Perform moving window FFT anomaly detection

winLength = anomalyLength #Length of moving window
winStep = int(round(anomalyLength / 5))

detectionCenterChannel = winLength / anomalyWavelength #Choose center channel of known anomaly frequency

detectionBandWidth = 2 # Half width of detection band (channels)
referenceChannelMin = 100
referenceChannelMax = 200

detectionBandMin = int(detectionCenterChannel - detectionBandWidth)
detectionBandMax = int(detectionCenterChannel + detectionBandWidth)

jj = 0

anomalyDetectionVec = pd.DataFrame(columns=['1', '2', '3', '4', '5', '6', '7'])

for ii in range(1, dataLength - winLength, winStep):
    winData = signalDataAnomaly[ii:(ii + winLength)]
    winFFT = abs(np.fft.fft(winData))

    anomalyArea = sum(winFFT[detectionBandMin:detectionBandMax]) / (detectionBandMax - detectionBandMin)
    referenceArea = sum(winFFT[referenceChannelMin:referenceChannelMax]) / (referenceChannelMax - referenceChannelMin)

    anomalyDetectionVec.loc[jj, '1'] = anomalyArea / referenceArea

    #Detected anomaly quotient
    anomalyDetectionVec.loc[jj, '2'] = ii

    #Moving window start index
    anomalyDetectionVec.loc[jj, '3'] = ii + winLength

    #Moving window end index
    anomalyDetectionVec.loc[jj, '4'] = np.floor(ii + winLength / 2)

    #Moving window center index
    anomalyDetectionVec.loc[jj, '5'] = time[ii]

    #Moving window start time
    anomalyDetectionVec.loc[jj, '6'] = time[ii + winLength]

    #Moving window end time
    anomalyDetectionVec.loc[jj, '7'] = time[int(np.floor(ii + winLength / 2))]

    #Moving window center time

    '''
    if True: #Set to TRUE for individual FFT plot
        anomalyHitFlag = sum(anomalyLabel[ii:(ii + winLength)]) > 0
        if anomalyHitFlag:
            plt.plot(winFFT(2:100), color='red')
        else:
            plt.plot(winFFT(2:100), color='blue')
        #drawnow
    '''

    jj = jj + 1

print('LOOP FINISHED')

jj = jj-1

print('PLOT ANOMALY DET')
plt.plot(anomalyDetectionVec['7'], anomalyDetectionVec['1'])

#todo ylim([-1 180])

print('PLOT ANOMALY LABEL')
plt.plot(time,(anomalyLabel*100), color='red')

#todo datetick

print('PLOT ANOMALY AGAIN')
plt.plot(anomalyDetectionVec['7'], anomalyDetectionVec['1'])
#todo ylim([-1 180])

print('PLOT ANOMALU LABEL AGAIN')
plt.plot(time,(anomalyLabel*100), color='red')
#todo datetick

#Calculate anomalyFlag by thresholding

anomalyFlagReference = anomalyLabel[anomalyDetectionVec['4']].astype(int)

truePositive = []
trueNegative = []
falsePositive = []
falseNegative = []
truePositiveRate = []
trueNegativeRate = []

print('START SECOND LOOP')
for ii in range(200):
    anomalyThreshold = ii
    #anomalyFlag = np.zeros(len(anomalyDetectionVec))
    anomalyFlag = 1 * (anomalyDetectionVec['1'] > anomalyThreshold)

    #Calculate confusion matrix


    totalSum = len(anomalyFlag)
    truePositive.append(sum(anomalyFlag & anomalyFlagReference))
    trueNegative.append(sum(~anomalyFlag & ~anomalyFlagReference))
    falsePositive.append(sum(anomalyFlag & ~anomalyFlagReference))
    falseNegative.append(sum(~anomalyFlag & anomalyFlagReference))
    truePositiveRate.append(np.array(truePositive).dot(np.linalg.pinv(np.expand_dims(
        np.array(truePositive).astype(int)+np.array(falseNegative).astype(int), axis=0))))

    trueNegativeRate.append(np.array(trueNegative).dot(np.linalg.pinv(np.expand_dims(
        np.array(falsePositive).astype(int)+np.array(trueNegative).astype(int), axis=0))))


plt.plot(truePositiveRate, color='green')
plt.plot(trueNegativeRate, color='red')

#Choose the threshold value when TRP and TNR crosses
diffTprTnr = abs(np.array(truePositiveRate) - np.array(trueNegativeRate))

anomalyThreshold = np.argmin(diffTprTnr)

anomalyFlag = 1 * (anomalyDetectionVec['1'] > anomalyThreshold)

#Calculate confusion matrix again for the choosen threshold
truePositive.append(sum(anomalyFlag & anomalyFlagReference))
trueNegative.append(sum(~anomalyFlag & ~anomalyFlagReference))
falsePositive.append(sum(anomalyFlag & ~anomalyFlagReference))
falseNegative.append(sum(~anomalyFlag & anomalyFlagReference))
truePositiveRate.append(np.array(truePositive).dot(np.linalg.pinv(np.expand_dims(
    np.array(truePositive).astype(int) + np.array(falseNegative).astype(int), axis=0))))

trueNegativeRate.append(np.array(trueNegative).dot(np.linalg.pinv(np.expand_dims(
    np.array(falsePositive).astype(int) + np.array(trueNegative).astype(int), axis=0))))

confMatrix = [[truePositive, falsePositive], [falseNegative, trueNegative]]
