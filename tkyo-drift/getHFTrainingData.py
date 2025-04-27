# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset

# ? If you are using a model on hugging face, you can use this utility to download the training data
# The data will be stored in you ~./cache folder
data_location = "SmallDoge/SmallThoughts"

def dataSetLoader (data_location):
    dataset = load_dataset("SmallDoge/SmallThoughts")
    print(dataset)
    return dataset

dataSetLoader(data_location)