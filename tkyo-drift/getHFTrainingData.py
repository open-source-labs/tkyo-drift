"""
Utility module for downloading and loading datasets from Hugging Face.
This module provides functionality to download training data from Hugging Face
datasets and store them in the local cache.
"""

# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
from datasets import load_dataset

# Default dataset to load
data_location = "SmallDoge/SmallThoughts"

def dataSetLoader(data_location):
    """
    Load a dataset from Hugging Face and store it in the local cache.
    
    This function downloads the specified dataset from Hugging Face and
    stores it in the user's ~/.cache folder. The dataset can then be used
    for training or evaluation purposes.
    
    Args:
        data_location (str): The Hugging Face dataset identifier (e.g., 'username/dataset-name')
        
    Returns:
        Dataset: The loaded Hugging Face dataset object
        
    Example:
        >>> dataset = dataSetLoader("SmallDoge/SmallThoughts")
        >>> print(dataset)
    """
    dataset = load_dataset("SmallDoge/SmallThoughts")
    print(dataset)
    return dataset

# Load the default dataset when the script is run directly
if __name__ == "__main__":
    dataSetLoader(data_location)