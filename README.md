# Steps to execute my code

## 1. Set up the python environment

    Python Version: 3.11.0
    The troch version I am using is compatible with python 3.11.
    I am not sure about other python versions, but 3.13.0 is not supported by torch

    I was using a python virtual environment
    requirement.txt details all the packages installed in that venv

## 2. Executing Preprocessing code

    The python code that performs preprocessing is inside "data" folder
    Please run the code with the current working directory
    So running preprocessing.py at the console should be "python data/preprocessing.py"

    The cleaned data will be stored in "data/2025-01-15/preprocessed/" including
        political science.csv
        psychology.csv
        sociology.csv

## 3. Executing the main program

    The main program, log files, models, and figures are stored in
    "FNN/",
    "FNN/logs/2025-01-??",
    "FNN/model/2025-01-??/??-??-??/",
    "FNN/model/2025-01-??/??-??-??/figures/"

    Running the main program in the console should be "python FNN/LinChungHsi_aifinal.py"
    Right after execution, importing torch if CUDA is present takes around a minute for my machine
    The main program is configured to train one 5-Fold cross validation
    The print outs on the console has limited information, the log file has all information during execution

    Filess created includes
        model_k.pth: the k models trained
        ??-??-??.log: the file logging all activities and metrics in the program
        simple_net: meta data to render the FNN's structure
        simple_net.png: the rendered image of the FNN's structure
