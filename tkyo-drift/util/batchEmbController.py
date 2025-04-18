# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True
# Import helper function to load and embed the data
import batchEmbWriter
from batchScalarWriteShared import write_shared_scalar_metrics


# Allows the use of time functions
import time
# JSON serialization/deserialization
import json
# Allow error logging for testing purposes
import traceback

def tkyoDriftSetTraining(data_set_Path, io_type, io_type_name):

    # Starts the total function timer
    startTotal = time.perf_counter()

    # Define the models
    MODELS = {
        'mini': 'sentence-transformers/all-MiniLM-L12-v2',
        'e5': 'intfloat/e5-base-v2',
    }


    print(f"Building scalar metrics for {io_type} as {io_type_name}.")
    write_shared_scalar_metrics(data_set_Path, io_type, io_type_name)

    # Iterate through models dictionary
    for model_type, model_name in MODELS.items():
        batchEmbWriter.trainingEmb(
            model_type=model_type,
            model_name=model_name,
            data_path=data_set_Path,
            io_type=io_type,
            io_type_name=io_type_name,
        )

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Full embedding run for {io_type} completed in: {endTotal - startTotal:.2f} seconds")

    return {"status": "ok", "message": "Training completed"}

# Checks that the file is run directly, not as an import
if __name__ == "__main__":
    # Error handling to check that there are 3 arguments and 1 script
    if len(sys.argv) != 4:
        # Print the error
        print(
            json.dumps(
                {
                    "error": "Usage: python3 tkyoDriftSetTraining.py <dataset_path> <io_type> <io_type_name>"
                }
            )
        )
        sys.exit(1)
    try:
        # assign the value of result to the evaluated result of invoking HNSW with the 3 input arguments
        result = tkyoDriftSetTraining(sys.argv[1], sys.argv[2], sys.argv[3])
        # Returns the value of result to javascript file
        print(json.dumps(result))
        # Catch all error handling
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
