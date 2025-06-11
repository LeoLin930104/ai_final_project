import pandas as pd                                         # Used for accessing the csv file and edit columns
import os                                                   # used for concat file directories
import re                                                   # Used for regular expressions
import nltk                                                 # Used for downloading stops word, etc (Natural Language Tookit)
from nltk.tokenize import word_tokenize                     # Used for tokenizing / splitting the abstracts into a list of words
from nltk.stem import WordNetLemmatizer, PorterStemmer      # Used to lemmatization

# Downloading necessary resources for nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')  
nltk.download('omw-1.4') 

def clean_abstracts(input_folder, output_folder, remove_list):
    """
    Description: 
        This is the function that performs preprocessing to the collected data.
    
    Arguments:
        input_folder: folder where the raw data are
        output_folder: folder to store the clean abstracts
        remove_list: lists of specific words or characters I want to remove from the abstracts
    """
    remove_pattern = r"|".join(map(re.escape, remove_list))                         # joins the punctuations into a map

    stop_words = set(nltk.corpus.stopwords.words('english'))                        # load english stop word from a library called nltk

    lemmatizer = WordNetLemmatizer()                                                # lemmatizer, which should have done lemmanization, but I couldn't get it to work
    stemmer = PorterStemmer()                                                       # So I used stemmer, which does lemmanization

    for file_name in os.listdir(input_folder):                                      # loops through the  files in the target folder
        if file_name.endswith('.csv'):                                              # Proceed if file extension is csv
            input_path = os.path.join(input_folder, file_name)                      # path to the file
            output_path = os.path.join(output_folder, file_name)                    # path to the file to export to

            try:
                df = pd.read_csv(input_path)                                        # read the csv with pandas
                if 'abstract' in df.columns:                                        # check if column "abstract" exists
                    df['abstract'] = (
                        df['abstract']
                        .str.lower()                                                # turn all word into lower case
                        .apply(lambda x: re.sub(r'<.*?>', '', x).strip())           # Remove HTML tags
                        .apply(lambda x: re.sub(remove_pattern, ' ', x).strip())    # Remove the punctuations specified
                        .apply(lambda x: re.sub(r'(?<=\w)–(?=\w)', ' ', x).strip()) # Strips "-" from words such as "now-a-days"
                        .apply(lambda x: re.sub(r'\d+', '', x))                     # Remove numbers
                        .apply(lambda x: re.sub(r'\s+', ' ', x).strip())            # Remove double spaces
                        .apply(lambda x: ' '.join([
                            stemmer.stem(word)                                      # Lemmatize the word for words in the split/tokenized abstract which are not stop words
                            for word in word_tokenize(x)                            # 
                            if word not in stop_words                               #
                        ]))
                    )

                    df = df[df['abstract'].apply(lambda x: len(x.split()) >= 40)]   # Remove abstracts that are too short, because they are often "abstract for this paper is unavailible                                               "
                    df = df.drop_duplicates(subset=['abstract'], keep='first')      # Remove Duplicates by the abstracts column
                    df = df.drop_duplicates(subset=['doi'], keep='first')           # Remove Duplicates by the DOI column

                os.makedirs(output_folder, exist_ok=True)                           # creats output directory if doesn't exist
                df.to_csv(output_path, index=False)                                 # Save the clean abstract into a csv file
                print(f"Processed and cleaned file: {output_path}")                 # Print when cleaning completes

            except Exception as e:                                                  # Error catching
                print(f"Error processing {input_path}: {e}")                        # Error printing


if __name__ == "__main__":
    """
    Description: 
        main function
    """
    folder = "data/2025-01-15"   # Folder of where the csv files resides
    output_folder = "data/2025-01-15/preprocessed"   # Folder to write out the cleaned abstracts

    # list of punctuations I found in the data that I want to remove
    remove_list = [ ',',  '.', '/', '$', '%', '@', ')', '(', '#', '\\', '!', '[', ']', '\'', '"', ':', ';', '?', '`', '‘', '’', '“', '”', '…', '–']

    clean_abstracts(folder, output_folder, remove_list)    # calls cleaning function
