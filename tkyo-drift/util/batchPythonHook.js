/**
 * Utility function to interface with Python batch processing scripts.
 * This module provides a bridge between Node.js and Python for batch processing
 * of training data and embeddings.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

// Full path to tkyoDriftSetTrainingHook.js
const __filename = fileURLToPath(import.meta.url);
// Directory containing the file (tkyo-drift)
const __dirname = path.dirname(__filename);

/**
 * Sets up and processes training data using Python batch processing scripts.
 * This function spawns a Python process to handle batch embedding generation
 * and training data setup.
 * 
 * @param {string} dataSetPath - Path to the dataset directory
 * @param {string} ioType - Type of input/output (e.g., 'input', 'output')
 * @param {string} ioTypeName - Name identifier for the I/O type
 * @returns {Promise<string>} The output from the Python process
 * @throws {Error} If the dataset path doesn't exist or if Python process fails
 */

export default async function tkyoDriftSetTraining(
  dataSetPath,
  ioType,
  ioTypeName
) {
  try {
    return new Promise((resolve, reject) => {
      // Creates a link between the data file and the initial function file
      const resolvedDataSetPath = path.resolve(process.cwd(), dataSetPath);

      // Check if the dataset folder exists
      if (!fs.existsSync(resolvedDataSetPath)) {
        // If not, throw an error
        throw new Error(
          `The dataSetPath "${resolvedDataSetPath}" does not exist.`
        );
      }
      // Ensures we are running tkyoDriftSetTraining.py correctly
      const scriptPath = path.join(__dirname, './batchEmbController.py');
      const pyProg = spawn('python3', [
        '-u',
        scriptPath,
        dataSetPath,
        ioType,
        ioTypeName,
      ]);

      let result = '';
      let error = '';

      // This function is for accepting to data from python
      // Data is the binary form of the result from python
      pyProg.stdout.on('data', (data) => {
        const chunk = data.toString();
        result += chunk;

        // Print each chunk as it comes in
        process.stdout.write(chunk); // or console.log(chunk) if you prefer line-based output
      });

      // This function is for error handling
      // Data is the binary from of the error from python
      pyProg.stderr.on('data', (data) => {
        // Error is the stringified version of the error from python
        error += data.toString();
      });

      pyProg.on('close', (code) => {
        if (code !== 0) {
          console.error('Python stderr:', error);
          reject(new Error(`Failed to embed training data`));
          return;
        }
        resolve(result);
      });
    });
  } catch (error) {
    throw new Error(
      `Error in readFromBin for the ${this.modelType} ${this.ioType} ${this.baselineType} model: ${error.message}`
    );
  }
}
