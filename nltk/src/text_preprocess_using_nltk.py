import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk                                     

nltk.download('stopwords')                      
nltk.download('punkt')                          
nltk.download('punkt_tab')
nltk.download('wordnet')                        
nltk.download('averaged_perceptron_tagger')     
nltk.download('averaged_perceptron_tagger_eng')

from nltk.corpus import stopwords               
from nltk.stem.porter import PorterStemmer      
from nltk.tokenize import word_tokenize         
from bs4 import BeautifulSoup                   
import re                                       
import string
import warnings
warnings.filterwarnings('ignore')


def test():
    print('This is Test')

# def main():
#     test()


if __name__ == "__main__":
    test()
