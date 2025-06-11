print("Loading Packages...")

import torch                                                # Imports the library that handles training, subseting Data, optimizer, Turn data into tensor objects
import torch.nn as nn                                       # Imports the module that the FNN inherits to initialize as a Neural Network object compatible with torch
import torch.nn.functional as F                             # Imports the module containing the softmax frunction, which is used at the output layer
from torch.utils.data import DataLoader                     # Imports the function in torch that manages subseting the data for k-fold
from torch.utils.data import TensorDataset                  # Imports the class in torch that holds the vectorize data and its label, so we can pass data (with labels) to the training function as one
from torchviz import make_dot                               # Imports the module that helps render the structure of the FNN into images
from sklearn.feature_extraction.text import TfidfVectorizer # Imports the TDIDF vectorizer
from sklearn.preprocessing import LabelEncoder              # Imports the label encoder that essentially encode [Psychology, Sociology, Political Science] as neuron 0, 1, 2 in the output layer
from sklearn.model_selection import KFold                   # Imports the module that give indices to help split data as K-folds
from sklearn.metrics import roc_curve                       # Imports the module that procudes ROC curve
from sklearn.metrics import roc_auc_score                   # Imports the module that calculated area under the curve for ROC curves
from sklearn.metrics import confusion_matrix                # Imports the module that produces the confusion matrix
from sklearn.metrics import precision_score                 # Imports the module that calculated precision score
from sklearn.metrics import recall_score                    # Imports the module that calculated recall score
from sklearn.metrics import f1_score                        # Imports the module that calculated f1 score
import seaborn as sns                                       # Imports the library that visualizes the confusion matrix produced
import matplotlib.pyplot as plt                             # Imported to display both the ROC curve and confussion matriix
import pandas as pd                                         # Imported to be used for Loading data from csv files
import os                                                   # Imported to create folders for storing log files and trained models
import logging                                              # Imported to log training status and progress
from datetime import datetime                               # Imported to create folder and log file names, I named each execution of the program with the time executed
import numpy as np                                          # Imported since some of the array objects uses numpy. Particular in ploting ROC curves and confussion matrices

def load_data(path, size):
    """
    Description: 
        The function that load in the three csv files of each field of study and return the concatinated pandas DataFramne
    
    Arguments:
        path: The path to the csv files
        size: The number of rows to read in from each csv files, the rows selected are random for each files
    
    Returns:
        data: the concatinated data as a pandas DataFrame object
    """
    fields = ["political science", "psychology", "sociology"]               # An array to store the fields of studies to be loaded in 
    data = []                                                               # Empty array to hold the data loaded temporarily
    for field in fields:                                                    # Loop through the fields
        data.append(pd.read_csv(path + field + '.csv').head(size))          # Pandas loads the csv as DataFrames and append the DF into datas
                                                                            # datas at this point has three DF objects 
    data = pd.concat(data)                                                  # Concatinate the three DF objects into one
    data.columns = ['doi', 'abstract', 'label']                             # Name the columns of the Dataframe
    return data                                                             # Return the data loaded

def data_vectorization(data, max_feature):
    """
    Description:
        This is the function that implements TFIDF, vectorizes the data, encode the label, then Package these data into a TensorDataset
    
    Arguments:
        data: A pandas DataFrame that has all the data the I want to vectorize.
        max_feature: maximum feature length results from TFIDF, 
                     feature length will be smaller than this number when the number unique terms in the corpus is smaller than this number,
                     I always choose a number smaller than the number unique terms in the corpus, max_feature is essentially feature length
    
    Returns:
        TensorDataset(X, y)
        max_feature: just passes the max_feature back, the reason is that I was testing using the number unique terms in the corpus as the feature length
                     which can be obtained in this function, so I pass max_feature back to be used in other functions.
    """
    vectorizer = TfidfVectorizer(max_features=max_feature)      # Initializing TFIDF Vectorizer with max_feature setting
    # vectorizer = TfidfVectorizer()                            # The code that would be used if I am testing with the number of unique terms in the corpus as feature length 
    X = vectorizer.fit_transform(data['abstract']).toarray()    # Vectorizing the abstract column of data
    max_feature = len(vectorizer.vocabulary_)                   # Extracting the max_feature in case I em using the commented out code
                    
    
    label_encoder = LabelEncoder()                              # Initializing label encoder
    y = label_encoder.fit_transform(data['label'])              # Encode the label column as y, so [Psychology, Sociology, Political Science] is represented as neuron 0, 1, 2 in the output layer
                                                                # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)                    # Parse the vectorized data into a tensor object
    y = torch.tensor(y, dtype=torch.long)                       # Parse the label into a tensor object
                                                                #
    return TensorDataset(X, y), max_feature, label_encoder      # Package the data and label into one dataset object and returns it along with max_feature and label encoder

class FNN(nn.Module):
    """
        Description:
            Here is where the FNN class is defined, defined meaning how many layers, size of layers represented with variable and more.
            This class inherits the nn module from torch.
    """
    def __init__(self, input_size, hidden_size, num_classes, dropout):
        """
        Description: 
            Here is where the definition of the FNN actually happens. 
            The FNN would first be initialized with the nn module from torch,
            then I would define the size of the layers with the layers size and input size: 
                A layer would be defined with the size itself and the size of the previous layer
        
        Arguments:
            input_size: The size of input layer, which is the size of the vectorzed data
            hidden_size: The size of the hidden layers, which is how many neurons does a layer has.
            num_classes: the number of labels, which is three, which is also the size of the output layer
            dropout: The dropout rate of the entire network
        
        Returns:
            self: which is the FNN object properly defined
        """
        super(FNN, self).__init__()                         # Contructor of the nn module
        self.fc1 = nn.Linear(input_size, hidden_size)       # First hidden layer: turn "input_size" neurons into "hidden_size" neurons.
                                                            # nn.Linear basically initializes the weight and bias in specified dimensions and packages into a callback
                                                            # This means what I put an array with correct dimensions into this callback-
                                                            # it would return an array computeded with the embedded weights and biases
        # self.fc2 = nn.Linear(hidden_size, hidden_size)    # Other hidden layers used during testing
        # self.fc3 = nn.Linear(hidden_size, hidden_size)    # 
        self.fc4 = nn.Linear(hidden_size, num_classes)      # Output layer: turn "hidden_size" neurons to 3 neurons
        self.relu = nn.ReLU()                               # Activation Function Relu
        self.dropout = nn.Dropout(dropout)                  # Storing the Dropout rate as an attribute
    
    def forward(self, x):
        """
        Description: 
            The function of the network that takes in the input and computes the output, the "feed-forward" part
            This function would take the layers, activation function, and dropout rate of this network and compute the input sequentially
            Sequentially being demonstrated in the comments below
        
        Arguments:
            x: the input to be computed, to be feed-forward
        
        Returns:
            x: the output after being computed
        """
        x = self.relu(self.fc1(x))      # input is put through the fc1(first layer) and then put through relu(activation function) 
                                        # essentially turn into the output of the first layer
        x = self.dropout(x)             # Then the dropout rate would deactivate(assign 0) some neurons from the output of the first layer
        # x = self.relu(self.fc2(x))    # Other hidden layers, operates identically as the first layer
        # x = self.dropout(x)           #
        # x = self.relu(self.fc3(x))    #
        # x = self.dropout(x)           #
        x = self.fc4(x)                 # The output of the first layer is then put through the output layer
        x = F.softmax(x, dim=1)         # The three neruons of the output layer is then normalized with the SoftMax function
        return x                        # Return the output layer as output
    
def render_structure(model, input_size):
    """
    Description: 
        This function would render the netowrk being passes in here
        The network is rendered through input dummy data which lets the library to know the structure of the netwrok
        Then the library would store the rendered structure as image file, which are what I have used in my report.
    
    Arguments:
        model: The neural network that is going to be rendered
        input_size: the size of the dummy data
    """
    dummy_input = torch.randn(1, input_size).to(device)             # Create dummy input with random values
    output = model(dummy_input)                                     # Get output from the model
    dot = make_dot(output, params=dict(model.named_parameters()))   # Create the structure of the nn through the dummy output
    dot.format = 'png'                                              # setting render image formate
    dot.render('simple_net')                                        # rendering and exporting image file

def kfold_cross_validation(device, tensor, label_encoder, hidden_size, max_feature, dropout, learning_rate, vector_len, dir, k=5, max_epochs=100, plot = True):
    """
    Description: 
        Here is where the training happens. A set of parameters will be inputted, and based on these hyperparameters, we train the model
        The training process involves using k-fold cross validation, and accuracies are going to be calculated and returned
    
    Arguments:
        device: The device (CPU or GPU) that is going to train the model
        tensor: The tensordataset that hold all the data and label that is going to be used for training
        label_encoder: Used here since confussion matrix and ROC curve needs the class names
        hidden_size: number of neurons in the hidden layer
        max_feature: size of the input layer
        dropout: Dropout rate of the network
        learning_rate: learning rate that the optimizer is going to multiply with the gradient
        vector_len: Number of data entries that is going to be used in training
        dir: the directory to save the model
        k: number of folds
        max_epochs: maximum amount of epochs training can undergo, this is set since for some configuration, training gets stuck
        plot: True of False for whether to plot confussion matrix and ROC curve or not
    Returns:
        metrics: Performance metrics of the k models 
    """
    kfold = KFold(n_splits=k, shuffle=True)     # Initializing the K-Fold module to split the data
    fold_metrics = []                           # Initializing empty array to store performance metrics
    input_size = max_feature                    # Renaming the feature length for more coherent code
    num_classes = 3                             # Number of classes, which is 3

    dir = os.path.join(os.getcwd(), "FNN/model/" + str(datetime.today().strftime('%Y-%m-%d/%H-%M-%S')) + "/")
    os.makedirs(dir + f"/figures/")             # create folder to store confusiom matrices and roc curves                            

    # Printing and logging hyperparamters
    print(f"Training with Vectorized Data {max_feature} features. Dropout Rate {dropout:.3f}. Dataset Size {3 * vector_len}. {hidden_size} neurons. Learning Rate {learning_rate:.5f}")
    logging.info(f"Training with Vectorized Data {max_feature} features. Dropout Rate {dropout:.3f}. Dataset Size {3 * vector_len}. {hidden_size} neurons. Leanring Rate {learning_rate:.5f}")
    

    for fold, (train_idx, val_idx) in enumerate(kfold.split(tensor)):               # Looping for k times, also acquiring the indices to split the data at each iteration 
        #print(f"Training for fold {fold+1}/{k}")                                   # Printing the start of training of current fold
        logging.info(f"Training for fold {fold+1}/{k}")                             # Logging  the start of training of current fold
        
        train_subsample = torch.utils.data.Subset(tensor, train_idx)                # Spliting the data into train and validation set with the KFold indices
        val_subsample = torch.utils.data.Subset(tensor, val_idx)
        
        train_loader = DataLoader(train_subsample, batch_size=32, shuffle=True)     # Initalize data loader for both training and validation set.
        val_loader = DataLoader(val_subsample, batch_size=32, shuffle=False)        # DataLoader will load data in batches, data shuffling happens here as well
        
        model = FNN(input_size, hidden_size, num_classes, dropout).to(device)       # Initialize the model and move the model to target device
        criterion = nn.CrossEntropyLoss()                                           # Initialize the module that has the Cross Entropy Loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)          # Initialize the optimizer to update the model with the calculated gradients of the loss function

        train_accuracies = []                                                       # Empty array to store training   accuracies for each epochs
        val_accuracies = []                                                         # Empty array to store validation accuracies for each epochs

        model.train()                                                   # Set model to training mode
        for epoch in range(max_epochs):                                 # Looping through 100 epochs, which is the maximum amount of possible epochs, however, early stopping kicks in mostly around 10 epochs
            correct_train, total_train = 0, 0                           # variables to calculate accuracies
            for inputs, labels in train_loader:                         # iterate through the training set in batches, seprating inputs and labels
                inputs, labels = inputs.to(device), labels.to(device)   # Putting the data on to the target device intended to be for training
                
                outputs = model(inputs)                                 # Get output from the FNN model by inputing the input data
                loss = criterion(outputs, labels)                       # calculate cross entropy loss
                
                # Compute training accuracy
                _, predictions = torch.max(outputs, 1)                  # Set prediction of the model as the neuron at the output layer that has the highest value
                correct_train += (predictions == labels).sum().item()   # Counting the correct predictions
                total_train += labels.size(0)                           # Updating the total amount of predictions made

                optimizer.zero_grad()                                   # as gradients calculated carries on to the next iteration of this loop, reseting gradients to zero
                loss.backward()                                         # calculates the loss of the network, or each weights and biases
                optimizer.step()                                        # Updates the model with the optimizer. Adam, instead of using SGD that just uses the current gradient, involves calculation with gradient of batches in the past.

            train_accuracy = correct_train / total_train                # Calculate accuracy on the training set
            train_accuracies.append(train_accuracy)                     # Stor the accuracy
            
            val_accuracy, _, _, _ = evaluate(model, val_loader, device) # Evaluate the model on the validation set
            val_accuracies.append(val_accuracy)                         # Store the accuracy

            # print(f"Epoch [{epoch+1}/{max_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")          # printing and logging training and validation set accuracy of the current epoch
            logging.info(f"Epoch [{epoch+1}/{max_epochs}], Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")     # logging so it is easier to track learning trend
            
            if((train_accuracy > 0.999                                              ) or    # My implementation of early stopping
               (train_accuracy > 0.995 and val_accuracy < max(val_accuracies)-0.003 ) or    # if accuracy on the training set exceeds 99.9%, the model will now start to overfit, terminate training
               (val_accuracy < max(val_accuracies)-0.015                            )       # if accuracy on the training set exceeds 99.5%, 
            ): break                                                                        #     and accuracy on the validation set droped 0.3% from the maximum validation accuracy in the past, overfitting, terminate training
                                                                                            # if accuracy on the validation set droped 1.5% from the maximum validation accuracy in the past, overfitting, terminate training
               

        val_accuracy, val_labels, val_preds, val_preds_probs = evaluate(model, val_loader, device) # Evaluating the trained model with the validation set, getting variables to calculate other metrics

        precision = precision_score(val_labels, val_preds, average='weighted', zero_division=0) # calculating precision considered as multiclass classification
        recall = recall_score(val_labels, val_preds, average='weighted', zero_division=0)       # calculating recall    considered as multiclass classification
        f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)               # calculating f1 score  considered as multiclass classification
        print(f"\tFold {fold+1} Validation Metrics - Accuracy: {val_accuracy:.4f}, Weighted Precision: {precision:.4f}, Weighted Recall: {recall:.4f}, Weighted F1: {f1:.4f}")

        precision = precision_score(val_labels, val_preds, average=None, zero_division=0)       # calculating precision considered as binary classification
        recall = recall_score(val_labels, val_preds, average=None, zero_division=0)             # calculating recall    considered as binary classification
        f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)                     # calculating f1 score  considered as binary classification
        for class_idx, class_name in enumerate(label_encoder.classes_):                         # logging performance metrix of each classes
            logging.info(f"  Class {class_name}: Precision: {precision[class_idx]:.4f}, Recall: {recall[class_idx]:.4f}, F1: {f1[class_idx]:.4f}")

        fold_metrics.append({           # Appending the calculated performance metric to a list as a dictionary
            'accuracy': val_accuracy,   # acurracy
            'precision': precision,     # precision
            'recall': recall,           # recall
            'f1': f1                    # f1 score
        })                              #

        # path = dir + "model_" + str(fold) + ".pth"    # format the filename of the model to save
        # torch.save(model, path)                       # save model
        #print("Save model to " + path)                 # print model saved to path
        # logging.info("Save model to " + path)         # log model saved to path

        plot_confusion_matrix(val_labels, val_preds, fold, k, label_encoder, dir, plot)    # Call function to plot confusion matrix
        plot_roc_curve(val_labels, val_preds_probs, fold, label_encoder, dir, plot)        # Call function to plot ROC curve

    metrics = {                                                 # Reshaping the performance metrics
        'accuracy': [m['accuracy'] for m in fold_metrics],      # accuracy
        'precision': [m['precision'] for m in fold_metrics],    # precision
        'recall': [m['recall'] for m in fold_metrics],          # recall
        'f1': [m['f1'] for m in fold_metrics],                  # f1 score
    }

    print("Average Validation Accuracy (Over", k,"folds): " + str(np.mean(metrics['accuracy'])))    # Print the average accuracies after finishing all k folds
    print("Confussion matrices and ROC curves are saved in: " + dir + "/figures/" + "\n")           # Informing the location where figures are saved
    logging.info("Confussion matrices and ROC curves are saved in: " + dir + "/figures/" + "\n")    # Logging the same statement


    return metrics  # return the performances metrics recorded

def evaluate(model, loader, device):
    """
    Description: 
        The function used to evaluate the model against a subset of the data. No training is performed here
    
    Arguments:
        model: the model that we want to evaluate, a FNN object
        loader: the data loader for the data subset we are evaluating the model with
        device: the device used compute the evaluation
    
    Returns:
        accuracy: the accuracy of the model's prediction on the data
        all_labels: a list of the actual label
        all_preds: a list of prediction make my the model
        all_preds_probs: a list storing the probabilities of all three label for every inputs
    """
    model.eval()                                                    # Setting the model to evaluation mode
    all_labels = []                                                 # Empty list for actualy labels
    all_preds = []                                                  # Empty list for predictions
    all_preds_probs = []                                            # Empty list for storing the raw values of the output layer, used to plot ROC curve
    correct, total = 0, 0                                           # Initializing variables to compute accuracy
    with torch.no_grad():                                           # Makes sure torch don't track gradient of the network, since tracking is only used for tuning, turning it off is more efficient
        for inputs, labels in loader:                               # iterate through the data in the data loader, which loads data in batches of 32, hence input"s" and label"s"
            inputs, labels = inputs.to(device), labels.to(device)   # moving the data to the device that is going to be used for evaluation

            outputs = model(inputs)                                 # getting the output layer of the model based on the input data
            _, predictions = torch.max(outputs, 1)                  # since its one hot encoding, the prediction is the index of the neuron with the highest value, or probability
            
            probs = outputs.cpu().numpy()
            all_preds_probs.extend(probs)                           # Store the probabilities of each labels

            total += labels.size(0)                                 # Update the total number of data predicted
            correct += (predictions == labels).sum().item()         # Update the number of predictions that are correct
            
            all_labels.extend(labels.cpu().numpy())                 # append the actually labels of this batch to the all_label list
            all_preds.extend(predictions.cpu().numpy())             # same, for appending to all_predict. And since "labels" and "predictions" are tensor objects, they are parsed into arrays
    accuracy = correct / total                                      # calculate accuracy based on the the formula (TP+TN) / (TP+TN+FP+FN) which is correct / total
    return accuracy, all_labels, all_preds, all_preds_probs         # return the accuracy evaluated, actual labels, and predicted labels which can be used to plot confussion matrices

def plot_confusion_matrix(labels, preds, fold, k, label_encoder, path, plot):
    """
    Description: 
        This function takes in a confussion matrix, in the form of numpy array, extract the information and plots a confussion matrix
    
    Arguments:
        labels: the actual label of data that is evaluated
        preds: predicted label of data that is evaluated
        fold: which fold is this confussion matrix of
        k: the total number of folds
        label_encoder: used to get the class names : [Psychology, Sociology, Political Science] as labels in the diagram
        path: directory to save the figure
        plot: a boolean to signal where to display the plot
    """
    
    cm = confusion_matrix(labels, preds)                                        # create the confussion matrix object with imported module
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100    # Turn the matrix values into percentages

    plt.figure(figsize=(7, 5))                                                  # setting figure size
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues',             # drawing the matrix as a heatmap so high percentage would be darker, better visualize the matrix
                xticklabels=label_encoder.classes_,                             # cm_percentage is then plotted on to this heatmap
                yticklabels=label_encoder.classes_)                             # labels on both axis are identical, which are the three class: [Psychology, Sociology, Political Science]
    plt.xlabel('Predicted Labels')                                              # label x axis as predicted
    plt.ylabel('True Labels')                                                   # label y axis as actual labels
    plt.title(f'Confusion Matrix (Fold {fold+1}/{k})')                          # Title the matrix with the correct fold number
    plt.savefig(path + f"/figures/conf_matrix_fold{fold+1}.png")                # save the diagram to directory
    if plot: plt.show()                                                         # display the confussion matrix if "plot" is true
    return                                                                      # returns nothing after window closed

def plot_roc_curve(labels, probs, fold, label_encoder, path, plot):
    """
    Description: 
        The function that plots ROC curves
    
    Arguments:
        labels: the actual label of data that is evaluated
        probs: the raw values from the output layer of data that is evaluated
        fold: which fold is this confussion matrix of
        k: the total number of folds
        label_encoder: used to get the class names : [Psychology, Sociology, Political Science] as labels in the diagram
        path: directory to save the figure
        plot: a boolean to signal where to display the plot
    Returns:

    """
    labels = np.array(labels)                                               # Turns labels and probs in a numpy array, 
    probs = np.array(probs)                                                 # since the module only takes in parameters in numpy array form, not list form
                                                     
    plt.figure(figsize=(10, 7))                                             # setting figure size
    for i in range(len(label_encoder.classes_)):                            # iterating through 0, 1, 2 as these are the classes, so plot each curve one at a time
        fpr, tpr, _ = roc_curve(labels == i, probs[:, i])                   # calculate the y, x axis of the roc curve, y and x being false postive rate and true positive rate
        auc_score = roc_auc_score(labels == i, probs[:, i])                 # calculate the area under the curve for one class
        plt.plot(                                                           # plot the ROC curve with the three information
            fpr,                                                            #   - plot the line with fpr and tpr
            tpr,                                                            #   - print the area under the curve as label to the curve
            label=f"{label_encoder.classes_[i]} (AUC = {auc_score:.2f})"
        )

    plt.xlabel("False Positive Rate")                                       # set x axis label as False Positive Rate
    plt.ylabel("True Positive Rate")                                        # set y axis label as True Positive Rate
    plt.title(f"ROC Curve (Fold {fold+1})")                                 # set Diagram Title
    plt.legend()                                                            # set legend
    plt.savefig(path + f"/figures/roc_curve_fold{fold+1}.png")              # save the diagram to directory
    if plot: plt.show()                                                     # display the ROC curves
    return                                                                  # returns nothing after closing the window

if __name__ == "__main__":
    """
    Description: 
        The main function of the program, here, several processes are happening
            1. Creating the log file that logs activity of the training
            2. Checking if a CUDA capable GPU is presents, configures the device that is going to be used for training
            3. load data from three csv files for each classes
            4. Create the variables of the network configuration that I am exploring the report
            5. Loop through different settings of each variables (if I am testing the variables)
            6. Train the model with K-Fold, which return performance metrics of the model
            7. Print the performance metric
    """
    try:
        log_dir = os.path.join(os.getcwd(), "FNN/logs/" + str(datetime.today().strftime('%Y-%m-%d/')))                  # directory of the folder to save the log file
        if not os.path.exists(log_dir):                                                                                 # create the directory is path doesn't exist
            os.makedirs(log_dir)                                                                                        #
        log_path = os.path.join(log_dir + str(datetime.today().strftime('%H-%M-%S')) + ".log")                          # directory of the log file
        logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')  # Create and configure the log file
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                           # Get the availible device to run the model on
        print("Using device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")                   # Print the availible device

        data_path = "data/2025-01-15/preprocessed/"        # Path to preprocessed data
        size = 800                                         # Number of rows to load from each csv files
        data = load_data(path = data_path, size = size)    # Load the data into a list

        dropout = 0.76836           # Optimal Dropout Rate          found in exploration
        max_feature = 1000          # Optimal feature length        found in exploration
        hidden_size = 218           # Optimal hidden layer length   found in exploration
        learning_rate = 0.00264376  # Optimal learning rate         found in exploration
        # number of layers          # I couldn't parameterize number of layers, the FNN class is defined optimally already
        
        # render_structure(FNN(max_feature, hidden_size, 3, dropout).to(device), max_feature)                             # renders the model with hyperparameters

        metrics = []     # Empty list to store performance metrics after training

        for i in range(0, 1):                                                       # Loop to use i to iterate though the range of hyperparameters that I want to test
            # size = int(800 / 10 * i)                                              # Uncomment to change rows of data
            # data = load_data(path = data_path, size = size)                       # Uncomment to load data in different "size"
            # dropout = i/20 + 0.01                                                 # Uncomment to change dropout rate
            # hidden_size = 2*i + 128                                               # uncomment to change layer size
            # learning_rate = 0.0041 * i + 0.0001                                   # Uncomment to change learning rate
            # max_feature = i * 50 + 500                                            # Uncomment to change feature length
            tensor, max_feature, label_encoder= data_vectorization(data, max_feature=max_feature) # Get TensorDataset which vectorizes the input and encodes the label
            metrics.append(kfold_cross_validation(device, tensor, label_encoder, hidden_size, max_feature, dropout, learning_rate, size, dir, k = 5, plot = False)) # Train Model and get performance metrics


        for k, metric in enumerate(metrics):                                                # loop through "metrics", where each "metric" contains the performance metrics for k model
            print(f"Model {k*5+1} - {(k+1)*5} : ")                                            # Print statements to print performance metrics of each in "metric"
            print("\tAccuracy of folds: \t[", end = "")                                     #
            [print(f"{x:.4f}", end = ", ") for x in metric['accuracy']]                     # Print accuracies for the k models
            print(f"]\tAverage accuracy: {np.mean(metric['accuracy']):.4f}")                # Print average accuracy for the k models
            for fold in range(len(metric['f1'])):                                           # Loop through the k folds
                print("\tMetrics for Fold", fold+1)                                         # Print current fold
                for class_idx, class_name in enumerate(label_encoder.classes_):             # Loop through classes
                    print("\t\t", class_name, ":\t", end = "")                              # Print class name
                    if class_idx > 0: print("\t", end = "")                                 # Alligning with '\t'
                    print(f"F1 score: {metric['f1'][fold][class_idx]:.4f}", end = "")       # Printing F1 score
                    print(f", Recall: {metric['recall'][fold][class_idx]:.4f}", end = "")   # Printing Recall score
                    print(f", Precision: {metric['precision'][fold][class_idx]:.4f}")       # Printing Precision score
            print()

    except Exception as e:                  # Error catching
        print(f"Error: {str(e)}")           # Printing error
        logging.info(f"Error: {str(e)}")    # logging error
