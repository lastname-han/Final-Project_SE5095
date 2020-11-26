# Final-Project_SE5095

My project is about machine learning-based fault detection of cutting machine via signal processing methods. 
The cutting machine deviates from its designed operating mechanism due to issues that occur within the cutting head of the process. 
During faulty-operation, the cutting machine produces a sound that is unique to what is causing the machine’s deviation from its design operating conditions. 
Using the machine’s tendency of sound variance, we can create a new strategy for blade replacement. 
Thus, goal of this project is to perform acoustic monitoring of the cutting machine, to automatically detect the machine defects. 

I collected audio data from cutting machine for the four trials at a healthy machine state (baseline) and three machine states with faulty operation (i.e., loose knife, no knife and no belt). 
Then I decomposed the audio signals through Fast Fourier Transform (FFT) and identified 8 key frequencies as the features.
I labeled the data with the 4 machine states. To classify the data, I used k-nearest neighbor algorithm (KNN). 
Basic classification data entails a predictor (or set of predictors) and a label. 
In our case, the set of predictors are the amplitudes of each 8 key frequency in audio signal and the labels are the four machine states (i.e. baseline, no knife, loose knife, no belt). 
The overall model of KNN was simply built using Scikit-learn library for Python. In the training process, k value which indicates the number of nearest neighbors was chosen as 5. 
In addition, I used confusion matrix to quantify the performance of the classifiers. 

In data file, rows represent 4 machine states and columns represent 8 key frequencies.
