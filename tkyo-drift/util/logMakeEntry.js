import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from './oneOffEmb.js';

export default function makeLogEntry(id, mathObject, type) {
  let logPath = '';
  // Construct the destination to the log in the data folder
  if (type === 'COS') {
    logPath = path.join(OUTPUT_DIR, 'logs', 'COS_log.csv');
  } else {
    logPath = path.join(OUTPUT_DIR, 'logs', 'EUC_log.csv');
  }

  // Create a timestamp
  const timestamp = new Date().toISOString();

  // Unpack keys from drift metrics: "semantic.problem.rolling"
  const grouped = {};

  // Dynamically unpack all similarity values
  for (const [key, value] of Object.entries(mathObject)) {
    // Split the key of the math object that got passed on each "." to get the model/ioType and baseline type
    const [modelType, ioType, baselineType] = key.split('.');
    // Create an object for the IO type within the grouped object
    if (!grouped[ioType]) grouped[ioType] = {};
    // Create an object for the model type within the ioType object within the grouped object
    if (!grouped[ioType][modelType]) grouped[ioType][modelType] = {};
    // Set the object's object's object's value to the COS similarity or EUC distance that got passed in
    grouped[ioType][modelType][baselineType] = value;
  }

  // Grab the model types from any incoming ioType
  const modelTypes = Object.keys(Object.values(grouped)[0]);

  // or dynamically detect if needed
  const baselineTypes = ['rolling', 'training'];

  const [[ioType, modelSet]] = Object.entries(grouped);

  const row = [id, timestamp, ioType];

  // Loop through each model in the group
  for (const model of modelTypes) {
    // loop through each baseline in the group
    for (const baseline of baselineTypes) {
      const val = modelSet[model]?.[baseline];
      row.push(val);
    }
  }

  const csvLine = row.join(',') + '\n';

  // Check if file exists
  const fileExists = fs.existsSync(logPath);

  // Write to file
  try {
    if (!fileExists) {
      // Set the headers that aren't dynamic
      const headerCols = ['ID', 'TIMESTAMP', 'I/O TYPE'];
      // For each model in the group
      for (const model of modelTypes) {
        // For each baseline in the group
        for (const baseline of baselineTypes) {
          // Add the dynamic headers to the array
          headerCols.push(
            `${model.toUpperCase()} ${baseline.toUpperCase()} ${type}`
          );
        }
      }
      // Delimit by comma... this is a CVS after all
      const headers = headerCols.join(',') + '\n';
      fs.writeFileSync(logPath, headers + csvLine);
    } else {
      fs.appendFileSync(logPath, csvLine);
    }
  } catch (error) {
    // * Something failed while writing the log
    // ? Could be disk permissions, file lock, etc.
    // ! Consider adding a fallback or alert
    console.error('Failed to write log entry:', error.message);
  }
}
