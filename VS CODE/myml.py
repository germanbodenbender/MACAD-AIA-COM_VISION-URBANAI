
import pandas as pd
import seaborn as sns
import altair as alt
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

sns.set(rc={'figure.figsize':(10,10)})

# Load data file
#CHANGE PATHS HERE TO MAKE IT WORK ON YOUR PC
data = pd.read_csv(r"D:\Google Drive\__ TRABAJO\IAAC-MACAD\3-AIA-COMPUTER VISION\AIA-COM_VISION-HOPS\AIA-COM_VISION-URBAN_BLOCK-FULL_SAMPLE.csv")
pd.options.display.max_columns = None

#COLUMNS THAT IM NOT USING ON THE TRAINED MODEL
data.drop(columns=[ "Plot_Min_Witdh", 'Plot_Count', 'Average_Build_Heigth', 'Gross_area', 'Radiation_Min'], inplace=True)
print(data.info())

#declare features
X = data.iloc[:,0:8]

print(X)

# Load and instantiate a StandardSclaer 
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler()

# Apply the scaler to our X-features
X_scaled = scalerX.fit_transform(X)

#declare regression target
y = data[["Radiation_Total", "Green_Comfort", "Energy_Balance"]]


from sklearn.preprocessing import MinMaxScaler
scalerY = StandardScaler()

y_scaled = scalerY.fit_transform(y)

"""**SPLIT INTO TRAIN AND TEST**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state = 21)

"""#Build model"""

# Instantiate a sequential model
model = tf.keras.models.Sequential()
n_cols = X_scaled.shape[1]  
# Add 2 dense layers of 50 and 32 neurons each
model.add(tf.keras.layers.Dense(64, input_shape=(n_cols,), activation='relu'))

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
  
# Add a dense layer with 1 value output
model.add(tf.keras.layers.Dense(3, activation= "linear"))
  
# Compile your model 
model.compile(optimizer = "adam", loss = "mean_absolute_error")

model.summary()

"""#Train model"""

# Fit your model to the training data for 200 epochs
#we assign this to history variable so we can plot the training data
history = model.fit(X_train,y_train,epochs=100, validation_split=0.2)

# summarize history for accuracy
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('loss function')
# plt.ylabel('mse')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# Evaluate your model accuracy on the test data
loss_test = model.evaluate(X_test,y_test)

# Print accuracy
print('mse_test:', loss_test)

#save model

model.save('COM_VISION-ML_REGRESSION_MODEL-V03.h5')
print("model saved")


# File path: CHANGE PATH HERE
filepath = 'COM_VISION-ML_REGRESSION_MODEL-V03.h5'


# Load the model
model = tf.keras.models.load_model(filepath, compile = True)
print("model loaded")

#____________________________hops function_________________________________________________



def predictions(Min_Depth, Max_Depth, Min_Heigth, Max_Heigth, Volume, Area, Green_area, FAR):

    samples_to_predict = [[Min_Depth, Max_Depth, Min_Heigth, Max_Heigth, Volume, Area, Green_area, FAR]]
    array_samples = np.array(samples_to_predict)
    scaled_samples = scalerX.transform(array_samples)

    # Generate predictions for samples
    predictions = model.predict(scaled_samples)
   
    # Decode them back
    final_predictions = scalerY.inverse_transform(predictions)
    #print(final_predictions)
    list1 = final_predictions.tolist()

    flat_list = []

    # iterating over the data
    for item in list1:
    # appending elements to the flat_list
        flat_list += item
    
    return flat_list




#Save a txt for grasshopper in case we need it for something
    np.savetxt("prediction.txt", final_predictions)
    print("TEXT SAVED")

