# Prevent _pycache_ creation, since these scripts only run on demand
import sys
sys.dont_write_bytecode = True

# Import helper function to create kmeans of data
import batchMakeKMeans

# This is good for vectors/matrices
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from datasets import Dataset, concatenate_datasets

# Allows the use of time functions
import time
import json
from datetime import datetime

# Access to OS functions
import os

# Allows for garbage collection
import gc

def trainingEmb(model_type, model_name, data_path, io_type, io_type_name):

    # Starts the total function timer
    startTotal = time.perf_counter()

    # Set device (MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # Load model components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    # Check if dataset exist
    if data_path:
        print(f"Loading all .arrow files from: {data_path}")

        arrow_files = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(".arrow")
        ]

        if not arrow_files:
            raise ValueError("No .arrow files found in the provided directory.")

        dataset_parts = [Dataset.from_file(fp) for fp in sorted(arrow_files)]
        dataset = concatenate_datasets(dataset_parts)
    else:
        raise ValueError(print("Could Not Retrieve Dataset"))

    # This prevents the creation of gradients
    @torch.no_grad()

    # When invoked, this will embed the current batch
    def embed_data(data):
        # Stores texts shorter than 512 tokens
        short_texts = []
        # Stores short text positions in the batch
        short_indices = []
        # Stores texts longer than 512 tokens
        long_texts = []
        # Stores long text positions in the batch
        long_indices = []

        # Split the input into short and long based on token length
        for index, text in enumerate(data):
            # Tokenizes each input
            tokenized = tokenizer.encode(text, add_special_tokens=False)
            # Check if the length of the token is greater than 512
            if len(tokenized) >= 512:
                # If greater, store text in long text
                long_texts.append(text)
                # Track index in long indices     
                long_indices.append(index)     
            else:
                # If less than 512, store text in short text
                short_texts.append(text)
                # Track index in short indices      
                short_indices.append(index)    

        # Creates an empty list to put the embeddings in the correct place
        embeddings = [None] * len(data)

        # Make sure there are short texts in the queue
        # If so, batch process all short texts
        if short_texts:
            # padding	Adds [PAD] tokens to match batch size
            # truncation	Cuts off long sequences (from the end)
            # max_length	Upper bound for number of tokens
            tokenized = tokenizer(
                short_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)
            # Vectorized values of the batched short embeddings
            output = model(**tokenized)
            short_embs = output.last_hidden_state.mean(dim=1).cpu().numpy()

            # Assign each embedding back to its original position
            for i, emb in zip(short_indices, short_embs):
                embeddings[i] = emb

        # Process long texts one by one (with chunking)
        for i, text in zip(long_indices, long_texts):
            # Chunk and embed, then assign
            embeddings[i] = embed_long_text(text)

        # Return embeddings in the same order as the input
        return np.stack(embeddings)
    
    # Handles the embeddings of a single long text
    def embed_long_text(text):
        chunks = chunk_text(text, tokenizer)
        tokenized = tokenizer(
            chunks,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)
        # Runs the tokenized chunks through the transformer model to get output embeddings
        output = model(**tokenized)
        # Averages across all chunk embeddings to a single vector
        return output.last_hidden_state.mean(dim=1).cpu().numpy().mean(axis=0)
    
    # Breaks the text up into overlapping chunks
    def chunk_text(text, tokenizer, max_length=512, stride=256):
        # Tokenizes each input
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Holds the tokenized chunks
        chunks = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i : i + max_length]
            if not chunk:
                break
            chunks.append(tokenizer.decode(chunk))
            if i + max_length >= len(tokens):
                break
        return chunks

    # Embed Data
    print(f"Embedding {io_type}s using {model_name}")
    # Initialize an empty list to store all input embeddings
    embeddings = []
    # Set the number of examples to process at once (smaller = less memory, larger = faster)
    batch_size = 10

    # Loop through the dataset in batches
    # range(start, stop, step) creates a sequence from 0 to len(dataset) in batch_size increments
    for i in range(0, len(dataset), batch_size):
        # Get a batch of input texts:
        # dataset[i : i + batch_size] slices the dataset to get current batch
        # [io_type_name] selects just the input column
        batch_raw = dataset[i : i + batch_size]

        # Use helper to resolve column names
        batch = resolve_io_column(batch_raw, io_type_name)

        # Convert the batch of texts to embeddings using our embedding function
        emb = embed_data(batch)

        # Compute and log scalar metrics for each item in batch
        # Loop through each embedded vector in the current batch
        for j, vector in enumerate(emb):
            timestamp = datetime.utcnow().isoformat() + "Z" 

            # ------------- << MODEL-SPECIFIC SCALAR METRICS >> -------------

            # L2 norm of the vector = magnitude of the embedding (model-dependent)
            norm = float(np.linalg.norm(vector))


            # Model-specific metrics
            model_metrics = {
                "norm": norm,
            }

            # Save each metric in its own file
            for metric, value in model_metrics.items():
                # Build file name: ioType.metric.modelType.training.scalar.jsonl
                file_path = f"tkyoData/scalars/{io_type}.{metric}.{model_type}.training.scalar.jsonl"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Write single metric with timestamp to file
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "timestamp": timestamp,
                        "metrics": {metric: value}
                    }) + "\n")

        # Add this batch's embeddings
        embeddings.append(emb)

        # Print progress every 10 batches
        if i % batch_size == 0:
            elapsed = time.perf_counter() - startTotal
            processed = min(i + batch_size, len(dataset))
            est_total = (elapsed / processed) * len(dataset) if processed else 0
            est_remaining = est_total - elapsed

            mins, secs = divmod(est_remaining, 60)
            if est_remaining < 60:
                eta_display = f"{int(secs)}s"
            else:
                eta_display = f"{int(mins):02d}:{int(secs):02d}"

            print(
                f"Processed {processed}/{len(dataset)} | ETA: {eta_display} ",
                end="\r",
                flush=True
            )


    print()

    embeddings = np.concatenate(embeddings)

    # Create data directories if they doesn't exist
    os.makedirs("tkyoData/vectors", exist_ok=True)
    #If you change the threshold below (currently 100,000), make sure you also change it on line 54 of pythonHNSW.py. Otherwise you will clip your training data set.
    if len(embeddings) < 100000:
        print(f"You have < 100000 {io_type} embeddings: Saving unfiltered embeddings to data directory.")
        # Assign the number of vectors for the training data
        num_vectors = embeddings.shape[0]

        # Assign the dimensions of each vector
        dims = embeddings.shape[1]

        # Create header bytes (8 bytes total)
        header_bytes = np.array([num_vectors, dims], dtype=np.uint32).tobytes()

        # Write to file (header first, then data)
        with open(f"tkyoData/vectors/{model_type}.{io_type}.training.bin", "wb") as f:

            # Write 8-byte header first
            f.write(header_bytes)

            # Then write the data
            embeddings.astype(np.float32).tofile(f)
    else:
        print(f"You have >=  100000 {io_type} embeddings: Performing K Means analysis to filter embeddings.")
        kMeansEmbedding = batchMakeKMeans.kMeansClustering(embeddings)

        # Assign the number of vectors for the training data
        num_vectors = kMeansEmbedding.shape[0]

        # Assign the dimensions of each vector
        dims = kMeansEmbedding.shape[1]

        # Create header bytes (8 bytes total)
        header_bytes = np.array([num_vectors, dims], dtype=np.uint32).tobytes()

        # Write to file (header first, then data)
        print(f"Writing KMeans centroids to disk.")
        with open(f"tkyoData/vectors/{model_type}.{io_type}.training.kmeans.bin", "wb") as f:

            # Write 8-byte header first
            f.write(header_bytes)

            # Then write the data
            kMeansEmbedding.astype(np.float32).tofile(f)

    # Ends timing for the entire function
    endTotal = time.perf_counter()
    print(f"Embedding completed in: {endTotal - startTotal:.2f} seconds")

    # -------- Final Cleanup: Release memory --------
    # Clear GPU cache and delete objects to free memory
    del model
    del tokenizer
    del dataset
    del dataset_parts
    del embeddings

    # Only runs if CUDA is used
    torch.cuda.empty_cache() 

    # Garbage Collect 
    gc.collect()

    return

def resolve_io_column(batch, io_type_name):
    try:
        # -------------------------------
        # Case 1: Flat column access
        # -------------------------------
        # This handles datasets where the column name is a top-level field like "input" or "output"
        # For example: Dataset({ "input": [...], "output": [...] })
        if hasattr(batch, "column_names") and io_type_name in batch.column_names:
            return list(batch[io_type_name])


        # -------------------------------
        # Case 1b: If we're passed a batch (dict of lists), like in a map() function
        # -------------------------------
        if isinstance(batch, dict) and io_type_name in batch:
            return list(batch[io_type_name])
        
        # -------------------------------
        # Case 2: Nested path access — supports expressions like ['conversations'][0]['value']
        # -------------------------------
        # This is common in OpenAI structured data, where each row is a nested dict
        #   Example:
        #   row = { 'conversations': [{'role': 'user', 'value': 'hi'}, {'role': 'assistant', 'value': 'hello'}] }
        #   io_type_name = "['conversations'][0]['value']"
        #
        # We dynamically create a lambda to access the field:
        #   accessor = lambda row: row['conversations'][0]['value']
        # Then apply it to every row in the batch.
        if "[" in io_type_name and "]" in io_type_name:
            accessor = eval(f"lambda row: row{io_type_name}")
            result = []

            for i, row in enumerate(batch):
                try:
                    val = accessor(row)  # Try to access the nested field
                    if val is not None:
                        result.append(val)
                except (KeyError, IndexError, TypeError):
                    # If something goes wrong (e.g., path doesn’t exist, value is None), skip it
                    print(f"[WARN] Skipping row {i}: missing nested path in {io_type_name}")
            return result

        # -------------------------------
        # Case 3: Fallback for dict-of-lists batch format
        # -------------------------------
        # Some HuggingFace DataLoaders return batches like:
        # {
        #   'input':  ['Hello', 'How are you?'],
        #   'output': ['Hi!',   'I am fine.']
        # }
        #
        # This block reconstructs row-wise records:
        #   [{'input': 'Hello', 'output': 'Hi!'}, ...]
        #
        # Then returns the value of the given field for each row.
        return [
            row[io_type_name]  # Access the target field in the reconstructed row
            for row in [
                {k: v[i] for k, v in batch.items()}  # Rebuild a row from column-wise format
                for i in range(len(next(iter(batch.values()))))  # Loop through index positions
            ]
        ]

    except Exception as e:
        # -------------------------------
        # Error Handling
        # -------------------------------
        # Catch any failure (invalid field, bad structure, etc.) and raise a meaningful error
        raise ValueError(f"Could not extract '{io_type_name}' from dataset: {e}")
