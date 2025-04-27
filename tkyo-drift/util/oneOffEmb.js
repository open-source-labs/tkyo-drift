/**
 * Core module for drift analysis using transformer-based embeddings.
 * This module provides the main pipeline for analyzing text drift using
 * multiple models and metrics, including cosine similarity and euclidean distance.
 */

import fs from 'fs';
import path from 'path';
import { v4 } from 'uuid';
import { DriftModel } from './oneOffModel.js';
import logMakeDriftEntry from './logMakeDriftEntry.js';
import makeErrorLogEntry from './logMakeError.js';
import logMakeInputEntry from './logMakeInputEntry.js';
import captureSharedScalarMetrics from './scalarCaptureShared.js';

/**
 * Available embedding models for drift analysis.
 * @type {Object.<string, string>}
 */

export const MODELS = {
  // t5: 'Xenova/sentence-t5-large',
  // bert: 'Xenova/sentence_bert', 
  mini: 'Xenova/all-MiniLM-L12-v2',
  e5: 'Xenova/e5-base-v2', 
};

/**
 * Root directory for all drift analysis data.
 * Contains subdirectories for vectors, scalars, and logs.
 * @type {string}
 */

export const OUTPUT_DIR = path.resolve('./tkyoData');

/**
 * Cache for pipeline output results to speed up model loading.
 * @type {Object}
 */

export const MODEL_CACHE = {};

/**
 * Main function for performing drift analysis on a text input.
 * 
 * This function orchestrates the entire drift analysis pipeline:
 * 1. Sets up necessary directories and validates model configuration
 * 2. Initializes and loads transformer models
 * 3. Generates embeddings for the input text
 * 4. Computes scalar metrics (both shared and model-specific)
 * 5. Saves embeddings and metrics to disk
 * 6. Calculates drift metrics (cosine similarity and euclidean distance)
 * 7. Logs all results
 * 
 * @param {string} text - The text to analyze for drift
 * @param {string} ioType - Type of input/output (e.g., 'input', 'output')
 * @returns {Promise<void>}
 * 
 * @throws {Error} If model configuration is invalid
 * @throws {Error} If there are issues with model construction or loading
 * 
 * @example
 * Analyze drift in an input text
 * await tkyoDrift("Sample text to analyze", "input");
 */

export default async function tkyoDrift(text, ioType) {
  // Stopwatch START 🏎️
  // console.time('Drift Analyzer Full Run');

  // Make model holder object, io types, baselines and directories (don't change these)
  const driftModels = {};
  const baselineTypes = ['rolling', 'training'];
  const subdirectories = ['vectors', 'scalars', 'logs'];

  //  ------------- << BEGIN try/catch Error Handling >> -------------
  // * Error handling is done within model method calls, which send the error to the catch block.
  try {
    //  ------------- << Make Directories >> -------------
    // Check if directory exists, if not, make it.
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    // Create subdirectories for vectors, scalars, and logs
    for (const dir of subdirectories) {
      const subdirPath = path.join(OUTPUT_DIR, dir);
      if (!fs.existsSync(subdirPath)) {
        fs.mkdirSync(subdirPath, { recursive: true });
      }
    }

    // Validate model config (we need the / and it's gotta be a string)
    for (const [type, name] of Object.entries(MODELS)) {
      if (typeof name !== 'string' || !name.includes('/')) {
        throw new Error(
          `Invalid or missing model ID for "${type}" model: "${name}"`
        );
      }
    }

    //  ------------- << Construct Model Combinations >> -------------
    try {
      // * For each model, for each baselineType, make a model and assign to driftModels object
      for (const [modelType, modelName] of Object.entries(MODELS)) {
        for (const baselineType of baselineTypes) {
          const key = `${modelType}.${ioType}.${baselineType}`;
          driftModels[key] = new DriftModel(
            modelType,
            modelName,
            ioType,
            baselineType
          );
        }
      }
    } catch (error) {
      throw new Error(
        `Error while constructing DriftModel objects: ${error.message}`
      );
    }
    //  ------------- << Initialize Model File Pathing >> -------------
    // * For each model, invoke set file path method
    // ! NOTE: If training data is not supplied, it will use the rolling file's path
    // Yes, this is intentional, check the ReadMe for why...
    for (const model of Object.values(driftModels)) {
      model.setFilePaths();
    }

    //  ------------- << Load the Xenova Models >> -------------
    // * Load all models sequentially
    // ! NOTE: Loading models sequentially is intentional, as they check the cache before attempting to load
    await Promise.all(
      Object.values(driftModels).map(async (model) => model.loadModel())
    );

    //  ------------- << Get Embeddings >> -------------
    // * Get embeddings for all inputs and outputs in parallel
    await Promise.all(
      Object.values(driftModels).map(async (model) => model.makeEmbedding(text))
    );

    //  ------------- << Get Scalar Metrics >> -------------
    // Capture shared scalar metrics once for each I/O type, for each baseline type
    captureSharedScalarMetrics(text, ioType);

    // * Calculate PSI values for scalar metric comparison
    await Promise.all(
      Object.values(driftModels).map(async (model) => {
        model.captureModelSpecificScalarMetrics(text);
      })
    );

    //  ------------- << Save Embedding Data >> -------------
    // * Save the embedding to the rolling/training files in parallel
    // ! NOTE: Write ops are done to separate files, this is safe
    await Promise.all(
      Object.values(driftModels).map(async (model) => model.saveToBin())
    );

    //  ------------- << Save Scalar Data >> -------------
    // * Save the embedding to the rolling/training files in parallel
    // Capture unique scalar metrics for each embedding model
    // ! NOTE: Write ops are done to separate files, this is safe
    await Promise.all(
      Object.values(driftModels).map(async (model) => model.saveScalarMetrics())
    );

    //  ------------- << Read Bin Files >> -------------
    // * Read up to N embeddings from binary blobs in parallel
    // ! NOTE: Read ops are non-blocking, this is safe
    // ? See Training Max Size/Rolling Max Size in ReadMe for more info
    // For each model, read from disk
    await Promise.all(
      Object.values(driftModels).map(async (model) => model.readFromBin())
    );

    //  ------------- << Get Baseline >> -------------
    // * Calculate Baseline values for each model in serial
    // For each model, calculate the baseline
    for (const model of Object.values(driftModels)) {
      model.getBaseline();
    }

    //  ------------- << Get Cosine Similarity >> -------------
    // * Calculate Cosine Similarity between input and baseline in serial
    const similarityResults = Object.fromEntries(
      Object.entries(driftModels).map(([key, model]) => [
        key,
        model.getCosineSimilarity(),
      ])
    );

    //  ------------- << Get Euclidean Distance >> -------------
    // * Calculate Euclidean Dist. between input and baseline in serial
    const distanceResults = Object.fromEntries(
      Object.entries(driftModels).map(([key, model]) => [
        key,
        model.getEuclideanDistance(),
      ])
    );

    //  ------------- << Make & Append Log Entries >> -------------
    // * Push the results to each log
    // Make shared ID and date for the cosine and Euclidean logs
    const sharedID = v4();
    logMakeDriftEntry(sharedID, similarityResults, 'COS');
    logMakeDriftEntry(sharedID, distanceResults, 'EUC');
    logMakeInputEntry(sharedID, text);
    
    //  ------------- << END try/catch Error Handling >> -------------
    // * Push any errors to the error log
    // ! NOTE: This platform intentionally fails silently
  } catch (error) {
    makeErrorLogEntry(error);
  }

  // Stopwatch END 🏁 (Comment this out in production)
  // console.timeEnd('Drift Analyzer Full Run');
}
