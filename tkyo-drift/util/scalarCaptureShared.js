/**
 * Utility functions for capturing and computing shared scalar metrics for text analysis.
 * These metrics include character length, entropy, word length, punctuation density,
 * and uppercase ratio, which are stored in JSONL files for drift analysis.
 */

import fs from 'fs';
import path from 'path';
import fsPromises from 'fs/promises';
import { config } from '../config.js';

/**
 * Captures and stores shared scalar metrics for a given text input.
 * The metrics are written to JSONL files in the scalars directory,
 * with separate files for each metric type.
 * 
 * @param {string} text - The text to analyze
 * @param {string} ioType - The type of input/output (e.g., 'input', 'output')
 * @returns {Promise<void>}
 */

export default async function captureSharedScalarMetrics(text, ioType) {
  const timestamp = new Date().toISOString();

  const metricSet = computeMetrics(text);
  // Write each metric to its respective file
  await Promise.all(
    Object.entries(metricSet).map(([metric, value]) => {
      // Construct the file path
      const filePath = path.join(
        config.outputDir,
        'scalars',
        `${ioType}.${metric}.rolling.scalar.jsonl`
      );

      // Ensure the parent directory exists, not the file itself
      const dirPath = path.dirname(filePath);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
      }

      const line =
        JSON.stringify({ timestamp, metrics: { [metric]: value } }) + '\n';
      return fsPromises.appendFile(filePath, line);
    })
  );
}

/**
 * Computes various scalar metrics for a given text string.
 * 
 * @param {string} text - The text to analyze
 * @returns {Object} An object containing the following metrics:
 *   - characterLength: Total number of characters
 *   - characterEntropy: Shannon entropy of character distribution
 *   - avgWordLength: Average length of words
 *   - punctuationDensity: Ratio of punctuation characters
 *   - uppercaseRatio: Ratio of uppercase letters
 */

function computeMetrics(text) {
  const metrics = {};

  // * Get Raw Input Length
  metrics.characterLength = text.length;

  // * Get Character Entropy
  const counts = {};
  for (const char of text) counts[char] = (counts[char] || 0) + 1;
  metrics.characterEntropy = -Object.values(counts).reduce((sum, count) => {
    const p = count / text.length;
    return sum + p * Math.log2(p);
  }, 0);

  // * Get Average Word Length
  const words = text.split(/\s+/);
  metrics.avgWordLength =
    words.length > 0
      ? words.reduce((sum, word) => sum + word.length, 0) / words.length
      : 0;

  // * Get Punctuation Density
  metrics.punctuationDensity =
    (text.match(/[.,!?;:]/g)?.length || 0) / text.length;

  // * Get Uppercase Ratio
  metrics.uppercaseRatio = (text.match(/[A-Z]/g)?.length || 0) / text.length;

  return metrics;
}
