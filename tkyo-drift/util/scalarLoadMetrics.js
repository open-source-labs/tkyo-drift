import fs from 'fs';
import readline from 'readline';
import path from 'path';
import { OUTPUT_DIR } from './oneOffEmb.js';

// * Function to read scalar metrics from the scalar jsonl files and group them by metric name
export async function loadScalarMetrics(
  metricNames,
  ioType,
  baselineType,
  // ! Note that most scalar metrics do not give a shit what model they come from, and only L2 Norm and Token Length do
  modelType = null,
  hybridMode = false
) {
  const metrics = {}; // this will hold the final merged metric data

  for (const metric of metricNames) {
    let filePath;

    // Configure file path based on model type first
    if (modelType) {
      filePath = path.join(
        OUTPUT_DIR,
        'scalars',
        // ? If the scalar metric is model specific, this will catch it (when this function gets invoked with a model value)
        `${ioType}.${metric}.${modelType}.${baselineType}.scalar.jsonl`
      );
    } else {
      filePath = path.join(
        OUTPUT_DIR,
        'scalars',
        // ? Otherwise, the scalar metric will come from a model agnostic file
        `${ioType}.${metric}.${baselineType}.scalar.jsonl`
      );
    }

    // handle hybrid mode if it's true
    if (hybridMode) {
      if (modelType) {
        filePath = path.join(
          OUTPUT_DIR,
          'scalars',
          // ? If the scalar metric is model specific, this will catch it (when this function gets invoked with a model value)
          `${ioType}.${metric}.${modelType}.rolling.scalar.jsonl`
        );
      } else {
        filePath = path.join(
          OUTPUT_DIR,
          'scalars',
          // ? Otherwise, the scalar metric will come from a model agnostic file
          `${ioType}.${metric}.rolling.scalar.jsonl`
        );
      }
    }

    // Skip if the file doesn't exist (can happen in partially populated environments)
    if (!fs.existsSync(filePath)) continue;

    // Create a line reader for the .jsonl file
    const fileStream = fs.createReadStream(filePath);
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity,
    });

    // Read each line from the file
    for await (const line of rl) {
      try {
        // Parse the JSON line (each line is a complete JSON object)
        const entry = JSON.parse(line);

        // Extract the scalar value from the "metrics" block
        const val = entry.metrics?.[metric];

        // Make sure it's a number before storing it
        if (typeof val === 'number') {
          if (!metrics[metric]) metrics[metric] = [];
          metrics[metric].push(val);
        }
      } catch (err) {
        console.warn(`Could not parse line in ${filePath}:`, err.message);
      }
    }
  }

  for (const metric in metrics) {
    const values = metrics[metric];

    if (hybridMode && baselineType === 'training') {
      // Use the first 10k lines from rolling files as a proxy for the the training data
      metrics[metric] = values.slice(0, 10000);
    }
    // use the most recent 1k lines from rolling
    if (baselineType === 'rolling') {
      metrics[metric] = values.slice(-1000);
    }
  }

  // returns a dictionary of arrays keyed by metric name
  return metrics;
}
