import fs from 'fs';
import path from 'path';
import { error } from 'console';
import fsPromises from 'fs/promises';
import { spawn } from 'child_process';
import { pipeline } from '@xenova/transformers';
import { OUTPUT_DIR, MODEL_CACHE } from './oneOffEmb.js';
import { fileURLToPath } from 'url';

export class DriftModel {
  constructor(modelType, modelName, ioType, baselineType) {
    this.baselineType = baselineType;
    this.modelType = modelType;
    this.modelName = modelName;
    this.ioType = ioType;
    this.distance = null;
    this.embedding = null;
    this.byteOffset = null;
    this.dimensions = null;
    this.vectorArray = null;
    this.scalarMetrics = null;
    this.baselineArray = null;
    this.embeddingModel = null;
    this.scalarFilePath = null;
    this.embeddingFilePath = null;
  }

  // * Function to set the file path
  setFilePaths() {
    try {
      // ?NOTE: training baselines may use KMeans files, which are handled inside the Python logic.
      // This JS path is not used for reading training data.

      // Construct the base file path for this model
      const baseName = `${this.modelType}.${this.ioType}.${this.baselineType}`;

      // Assemble the embedding file path (.bin file)
      const vectorPath = path.join(OUTPUT_DIR, 'vectors', `${baseName}.bin`);
      const vectorKmeansPath = path.join(
        OUTPUT_DIR,
        'vectors',
        `${baseName}.kmeans.bin`
      );
      const fallbackPath = path.join(
        OUTPUT_DIR,
        'vectors',
        `${this.modelType}.${this.ioType}.rolling.bin`
      );

      // Use rolling file path if there is no training data.
      // ? ctrl+f the README for hybrid mode if you want to know why
      this.embeddingFilePath = fs.existsSync(vectorKmeansPath)
        ? vectorKmeansPath
        : fs.existsSync(vectorPath)
        ? vectorPath
        : fallbackPath;

      // Scalar metric path (.scalar.jsonl)
      this.scalarFilePath = path.join(
        OUTPUT_DIR,
        'scalars',
        `${baseName}.scalar.jsonl`
      );
    } catch (error) {
      throw new Error(
        `Error in setFilePath for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to load the embedding model
  async loadModel() {
    try {
      // Don't reload a model if it's loaded.
      if (this.embeddingModel) return;

      // Check the global cache to see if the model was already downloaded
      if (MODEL_CACHE[this.modelName]) {
        this.embeddingModel = await MODEL_CACHE[this.modelName];
        return;
      }

      // Load the model using xenova transformer and the model ID
      this.embeddingModel = await pipeline(
        'feature-extraction',
        this.modelName
      );
      MODEL_CACHE[this.modelName] = this.embeddingModel;
    } catch (error) {
      throw new Error(
        `Error in loadModel for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to make an embedding from an input/output pair
  async makeEmbedding(text) {
    try {
      // Validate that the text is not null/undefined/empty
      if (typeof text !== 'string' || text.trim() === '') {
        throw new Error(
          'Expected a non-empty string but received invalid input.'
        );
      }

      // Invoke the load model if it hasn't been done yet
      await this.loadModel();

      // Tokenize the text to check length
      const tokens = await this.embeddingModel.tokenizer(text);
      const tokenCount = tokens.input_ids.size;

      const maxLength = 512;
      const stride = 256;

      if (tokenCount < maxLength) {
        // Short Text found: embed normally
        const result = await this.embeddingModel(text, {
          pooling: 'mean',
          normalize: false,
        });

        // Save embedding to the object
        this.embedding = result.data;
      } else {
        // Long text found, embed each, and then average
        const chunks = [];

        for (let i = 0; i < tokenCount; i += stride) {
          const chunkIds = tokens.input_ids.data.slice(i, i + maxLength);

          if (chunkIds.length === 0) break;

          const chunkText = this.embeddingModel.tokenizer.decode(chunkIds, {
            skip_special_tokens: true,
          });

          const result = await this.embeddingModel(chunkText, {
            pooling: 'mean',
            normalize: true,
          });

          chunks.push(result.data);

          if (i + maxLength >= tokenCount) break;
        }

        // Average all chunk embeddings
        const dim = chunks[0].length;
        const avg = new Float32Array(dim);

        for (let i = 0; i < chunks.length; i++) {
          for (let j = 0; j < dim; j++) {
            avg[j] += chunks[i][j];
          }
        }

        for (let j = 0; j < dim; j++) {
          avg[j] /= chunks.length;
        }

        // Save embedding to the object
        this.embedding = avg;
      }

      // Check if result.data exists and is a numeric array
      if (!(this.embedding instanceof Float32Array)) {
        throw new Error('Embedding result is not a valid Float32Array.');
      }
      // Check if the embedding is empty
      if (this.embedding.length === 0) {
        throw new Error('Embedding array is empty.');
      }

      // Save dimensions to object (the actual vector dim is at position 1)
      this.dimensions = this.embedding.length;

      // save byte offset to object
      this.byteOffset = this.embedding.byteOffset;
    } catch (error) {
      throw new Error(
        `Error in makeEmbedding for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to Save Data to file path
  async saveToBin() {
    // Skip if training — this method is only for rolling baseline
    if (this.baselineType === 'training') return;

    try {
      // Make a new float 32 array out of the embedding
      const float32Array = new Float32Array(this.embedding);

      // Check to make sure the embedding contains only numbers
      if (
        !float32Array.length ||
        float32Array.some((v) => typeof v !== 'number' || Number.isNaN(v))
      ) {
        throw new Error('Invalid embedding: contains non-numeric values.');
      }

      // Buffer the vector
      const embeddingBuffer = Buffer.from(float32Array.buffer);

      // Check if the file exists
      const fileExists = fs.existsSync(this.embeddingFilePath);

      // Validate embedding dimensions BEFORE writing
      if (float32Array.length !== this.dimensions) {
        throw new Error(
          `Dimension mismatch: embedding has ${float32Array.length} values, expected ${this.dimensions}`
        );
      }

      // If the file doesn't exist, add the vector, and write a new header
      if (!fileExists) {
        // Allocate space for the header shit
        const headerBuffer = Buffer.alloc(8);

        // Total vectors is 1 on first write
        headerBuffer.writeUInt32LE(1, 0);

        // Update the header with vector dimensions
        headerBuffer.writeUInt32LE(this.dimensions, 4);

        // Concatenate the header data with the vector data
        const fullBuffer = Buffer.concat([headerBuffer, embeddingBuffer]);

        // Write the header to the file
        await fs.promises.writeFile(this.embeddingFilePath, fullBuffer);

        // If the file does exist, append the vector, and update the existing header
      } else {
        // Validate the file header matches this.dimensions BEFORE writing
        const fd = await fs.promises.open(this.embeddingFilePath, 'r');
        const headerBuffer = Buffer.alloc(8);
        await fd.read(headerBuffer, 0, 8, 0);
        await fd.close();

        const fileVectorDims = headerBuffer.readUInt32LE(4);
        if (fileVectorDims !== this.dimensions) {
          throw new Error(
            `File dimension mismatch: file expects ${fileVectorDims}, embedding has ${this.dimensions}`
          );
        }

        // Append new vector
        await fs.promises.appendFile(this.embeddingFilePath, embeddingBuffer);

        // Recalculate new vector count
        const stats = await fs.promises.stat(this.embeddingFilePath);
        const vectorsInBinCount = Math.floor(
          (stats.size - 8) / (this.dimensions * 4)
        );

        // Update header: numVectors
        const fullHeaderBuffer = Buffer.alloc(8);
        fullHeaderBuffer.writeUInt32LE(vectorsInBinCount, 0);
        fullHeaderBuffer.writeUInt32LE(this.dimensions, 4);

        const fileHandle = await fs.promises.open(this.embeddingFilePath, 'r+');
        // Rewrite the full header 8 bytes to prevent bin corruption
        // ! Once upon a time, we only updated the first 4 bytes. It broke everything.
        await fileHandle.write(headerBuffer, 0, 8, 0);
        await fileHandle.close();
      }
    } catch (error) {
      throw new Error(
        `Error in saveToBin for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to read the contents of the Bins, Build an HNSW
  async readFromBin() {
    // Full path to DriftModel.js
    const __filename = fileURLToPath(import.meta.url);
    // Directory containing the file (util)
    const __dirname = path.dirname(__filename);

    // Creates a link between the data file and the inital function file
    const resolvedDataSetPath = path.resolve(
      process.cwd(),
      this.embeddingFilePath
    );

    // Check if the dataset folder exists
    if (!fs.existsSync(resolvedDataSetPath)) {
      // If not, throw an error
      throw new Error(
        `The dataSetPath "${resolvedDataSetPath}" does not exist.`
      );
    }
    // Ensures we are running pythonHNSW.py correctly
    const scriptPath = path.join(__dirname, 'pythonHNSW.py');

    try {
      return new Promise((resolve, reject) => {
        const pyProg = spawn('python3', [
          scriptPath,
          this.ioType,
          this.modelType,
          JSON.stringify(Array.from(this.embedding)),
          this.baselineType,
          this.embeddingFilePath,
        ]);

        let result = '';
        let error = '';

        // This function is for accepting to data from python
        // Data is the binary form of the result from python
        pyProg.stdout.on('data', (data) => {
          // Result is the stringified version of the result from python
          result += data.toString();
        });

        // This function is for error handling
        // Data is the binary from of the error from python
        pyProg.stderr.on('data', (data) => {
          // Error is the stringified version of the error from python
          error += data.toString();
        });

        pyProg.on('close', (code) => {
          if (code !== 0) {
            reject(
              new Error(
                `Python process failed in readFromBin for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error}`
              )
            );
            return;
          }

          // Destructure from result after parsing the result, changing it from a string to an object
          const { centroids, distances } = JSON.parse(result);

          // Assign the centroids to vectorArray
          this.vectorArray = centroids;

          // Assign the average of all distances from centroids to the distance
          this.distance =
            // Since distance is null occasionally, we only assign if it isn't
            // ! Python is not returning typed arrays currently, so we do not need to check for instance of float32Array
            Array.isArray(distances) && distances.length > 0
              ? distances.reduce((sum, val) => sum + val, 0) / distances.length
              : null;
          resolve({ centroids, distances });
        });
      });
    } catch (error) {
      throw new Error(
        `Error in readFromBin for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to get baseline value from vectorArray
  getBaseline() {
    try {
      // Check to make sure the vectorArray was correctly set in readFromBin
      // ! Python is not returning typed arrays currently, so we do not need to check for instance of float32Array
      if (!Array.isArray(this.vectorArray) || this.vectorArray.length === 0) {
        throw new Error('Baseline vectorArray is missing or empty.');
      }

      // Validate the structure of the vector array before attempting to reduce it
      // ! Python is not returning typed arrays currently, so we do not need to check for instance of float32Array
      if (!Array.isArray(this.vectorArray[0])) {
        throw new Error('Baseline vectorArray is not an array of arrays.');
      }

      // If readFromBin returns a single vector, skip the math and return out.
      if (this.vectorArray.length === 1) {
        this.baselineArray = new Float32Array(this.vectorArray[0]);
        return;
      }

      // Set the baseline array to the proper dimensions
      this.baselineArray = new Float32Array(this.dimensions);

      // Set each value in the baseline array equal to the mean of the vector array
      for (let i = 0; i < this.dimensions; i++) {
        this.baselineArray[i] =
          this.vectorArray.reduce(
            (accumulator, currentValue) => accumulator + currentValue[i],
            0
          ) / this.vectorArray.length;
      }

      // Sanity check: Make sure the baseline is valid
      const valid = this.baselineArray.every(
        (val) => typeof val === 'number' && !Number.isNaN(val)
      );

      if (!valid || this.baselineArray.length !== this.dimensions) {
        throw error(
          'Error getting baseline: invalid values or dimension mismatch'
        );
      }
    } catch (error) {
      throw new Error(
        `Error in getBaseline for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to get cosine similarity between baseline and embedding
  getCosineSimilarity() {
    try {
      // Validate the embedding and baselines both exist
      if (
        !(this.embedding instanceof Float32Array) ||
        !(this.baselineArray instanceof Float32Array)
      ) {
        throw new Error('Missing embedding or baseline for cosine similarity.');
      }

      // Validate that both the baseline and embedding lengths match
      if (this.embedding.length !== this.baselineArray.length) {
        throw new Error(
          `Embedding and baseline length mismatch: ${this.embedding.length} vs ${this.baselineArray.length}`
        );
      }

      // Normalize both vectors to unit length
      const normalize = (vec) => {
        const mag = Math.sqrt(vec.reduce((sum, v) => sum + v * v, 0));
        return vec.map((v) => v / mag);
      };

      const a = normalize(this.embedding);
      const b = normalize(this.baselineArray);

      // Calculate the dot product of the A and B arrays
      let dotProduct = 0;
      for (let i = 0; i < this.dimensions; i++) {
        dotProduct += a[i] * b[i];
      }

      return dotProduct; // Math.min(1, Math.max(-1, dotProduct));
    } catch (error) {
      throw new Error(
        `Error in getCosineSimilarity for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to calculate the euclidean distance from the baseline
  getEuclideanDistance() {
    try {
      // Validate that the embedding and baselines exist
      if (
        !(this.embedding instanceof Float32Array) ||
        !(this.baselineArray instanceof Float32Array)
      ) {
        throw new Error('Missing embedding or baseline.');
      }

      // Validate that the embedding and baselines are the same length
      if (this.embedding.length !== this.baselineArray.length) {
        throw new Error(
          `Embedding and baseline length mismatch: ${this.embedding.length} vs ${this.baselineArray.length}`
        );
      }

      // If distance was already computed by Python, use it...
      if (typeof this.distance === 'number') {
        return this.distance;
      }

      // ...otherwise, calculate the distance between the embedding and baselineArray
      return Math.sqrt(
        this.embedding.reduce(
          (sum, a, i) => sum + (a - this.baselineArray[i]) ** 2,
          0
        )
      );
    } catch (error) {
      throw new Error(
        `Error in getEuclideanDistance for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to siphon PSI distribution metrics
  captureModelSpecificScalarMetrics(text) {
    try {
      // Skip if training — this method is only for rolling baseline
      if (this.baselineType === 'training') return;
      // Calculate vector L2 norm
      const norm = Math.sqrt(
        this.embedding.reduce((sum, val) => sum + val * val, 0)
      );

      this.scalarMetrics = {
        timestamp: new Date().toISOString(),
        metrics: {
          norm,
        },
      };
    } catch (error) {
      throw new Error(
        `Error in extractModelScalarMetrics for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }

  // * Function to write model-specific scalar metrics to separate files
  async saveScalarMetrics() {
    // Skip if training — this method is only for rolling baseline
    if (this.baselineType === 'training') return;

    try {
      // Create a timestamp for this scalar metric entry
      const timestamp = new Date().toISOString();

      // Unpack the scalarMetrics object into individual [metric, value] pairs
      // For example: { norm: 11.4, tokenLength: 128 } → [['norm', 11.4], ['tokenLength', 128]]
      const entries = Object.entries(this.scalarMetrics.metrics);

      // For each metric, write its value to a separate file
      await Promise.all(
        entries.map(async ([metric, value]) => {
          // Construct the file path using: ioType.metric.modelType.baselineType.scalar.jsonl
          // Example: input.norm.semantic.rolling.scalar.jsonl
          const filePath = path.join(
            OUTPUT_DIR,
            'scalars',
            `${this.ioType}.${metric}.${this.modelType}.rolling.scalar.jsonl`
          );

          // Format the line as a JSONL object with timestamp and single metric
          const line =
            JSON.stringify({
              timestamp,
              metrics: { [metric]: value },
            }) + '\n';

          // Append the scalar entry to the file
          await fsPromises.appendFile(filePath, line);
        })
      );
    } catch (error) {
      // If anything fails (e.g., write error, path issue), log and rethrow
      throw new Error(
        `Error in saveScalarMetrics for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
      );
    }
  }
}
