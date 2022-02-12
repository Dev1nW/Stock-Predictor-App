import tkinter as tk
from tkinter import filedialog, Text
import os
from datetime import datetime
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf 
from tensorflow import keras
import math
import time


root = tk.Tk()
Stocks = []
#Holds array of Stocks 


#Load data from save.txt file if any 
if os.path.isfile('save.txt'):
    with open('save.txt', 'r') as f:
        tempStock = f.read()
        tempStock = tempStock.split(',')
        Stocks = [x for x in tempStock if x.strip()]

#Add Stock to Stocks 
def add_Stock():
    
    for widget in frame.winfo_children():
        widget.destroy()
    
    filename = filedialog.askopenfilename(initialdir ='/', title='Select File', filetypes=(("CSV","*.csv"), ("all files", "*.*")))
    #Open file explorer, look for csv files
    
    if filename != '':
        Stocks.append(filename)
    #Since you can open the window explorer without choosing a file 
    #Only consider meaningful files

    for Stock in Stocks:
        label = tk.Label(frame, text='{}'.format(Stock_name), bg='gray')
        label.pack()
    #This creates the label which is displayed on the App itself
        

def run_Stock_Analysis():
    #run Machine Learning Algorithm for each Stock csv file
    for Stock in Stocks:

        def Create_model(dataset, x_train):
            model = keras.models.Sequential()
            model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(keras.layers.LSTM(50, return_sequences=False))
            model.add(keras.layers.Dense(25))
            model.add(keras.layers.Dense(1))

            model.compile(optimizer='adam',loss='mean_squared_error')

            return model
      
        df = pd.read_csv(Stock)
        data = df.filter(['Adj Close'])
        dataset = data.values
        csv_dir = Stock
        Stock_dir = csv_dir.replace('.csv', '') 
        #We obtain the Stock name so we can create a folder which will hold 2 things
        #1. Model information
        #2. Graphs of the results

        print('\n\nInformation for stock will be saved to: {}\n\n'.format(Stock_dir))

        training_data_len = math.ceil(len(dataset)*.8)

        #Scale data

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        train_data = scaled_data[0:training_data_len, :]
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
        
        model = Create_model(dataset, x_train)

        #Create path for the model checkpoints to be saved to

        #checkpoint_path = "{}/cp.ckpt".format(Stock_name)
        #checkpoint_dir = os.path.dirname(checkpoint_path)

        #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True,verbose=1)

        model.fit(x_train, y_train, batch_size=1, epochs=20)#for checkpoints add ', callbacks=[cp_callback])'

        model.save('{}/model'.format(Stock_dir))

        #Run for 20 Epochs each instance

        test_data = scaled_data[training_data_len-60:, :]

        x_test = []
        y_test = dataset[training_data_len:, :]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
    
        x_test = np.reshape(x_test,(x_test.shape[0], x_test.shape[1], 1))

        pred = model.predict(x_test)

        pred = scaler.inverse_transform(pred)

        #os.listdir(checkpoint_dir)

        rmse=np.sqrt(np.mean(((pred- y_test)**2)))

        print("\n\nmodel root mean squared error: {:5.2f}%\n\n".format(rmse))  

        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = pred

        if os.path.isdir(Stock_dir):
            Stock_name = os.path.basename(Stock_dir)

        #Get only the name of the stock and not the path

        fig = plt.figure(figsize=(16,8))
        plt.title('Model for {}'.format(Stock_name))
        plt.xlabel('Number of days', fontsize=18)
        plt.ylabel('Adj Close Price', fontsize=18)
        plt.plot(train['Adj Close'])
        plt.plot(valid[['Adj Close', 'Predictions']])
        plt.legend(['Train', 'Val','Predictions'],loc = 'lower right')

        #Create Graph with Stock information
        
        print("Would you like to save the graph:\n")
        print("1 - Save graph and go to next stock\n")
        print("2 - Don't save graph and go to next stock\n")

        fig.canvas.draw()

        usr_input = input()

        if usr_input == '1':
            fig.show()
            fig.savefig("{}/graph.png".format(Stock_dir))
            #Save graph of results
            plt.pause(5.)
            #Show the graph for roughly 5 seconds
            plt.close(fig)
        else:
            fig.show()
            plt.pause(5.)
            #show the graph for roughly 5 seconds
            plt.close(fig)

        

        
def remove_Stocks():
    print("Removing Stocks Now.")
    try:
        os.remove('./save.txt')
        Stocks.clear()
    #remove the save.txt and stocks from stocks
    except:
        Stocks.clear()
    #clear the list of Stocks
    
    for widget in frame.winfo_children():
        widget.destroy()
    #remove the pathnames from Stock
    
    print("Stocks Removed.\n")

if __name__ =='__main__':
    canvas = tk.Canvas(root, height=700, width=700, bg='#263D42')
    canvas.pack()
    #create the window that will open

    frame = tk.Frame(root, bg='white')
    frame.place(relwidth=.8, relheight=.8, relx =0.1, rely=0.1)
    #create a frame, this is where the file/pathname inforamtion will be displayed

    openStock = tk.Button(root, text ='Choose Stock .csv File', padx = 10, pady=5, fg='white', bg='#263D42', command=add_Stock)
    openStock.pack(side="top", fill='both', expand=True, padx=4, pady=4)
    #Create a button to open the file explorer

    RunAnalysis = tk.Button(root, text ='Run Stock Analysis', padx = 10, pady=5, fg='white', bg='#263D42', comman=run_Stock_Analysis)
    RunAnalysis.pack(side="top", fill='both', expand=True, padx=4, pady=4)
    #Create a button to run the programs

    RemoveStocks = tk.Button(root, text ='Remove Stocks', padx = 10, pady=5, fg='white', bg='#263D42', comman=remove_Stocks)
    RemoveStocks.pack(side="top", fill='both', expand=True, padx=4, pady=4)
    #Create a button to remove the Stocks

    for Stock in Stocks:
        label = tk.Label(frame, text=Stock)
        label.pack()
    #Create labels to show programs in the save.txt file
        
    root.mainloop()

    with open('save.txt', 'w') as f:
        for Stock in Stocks:
            f.write(Stock + ',')
        #write all the programs in a file so that you can use them later, even if you close the window
            