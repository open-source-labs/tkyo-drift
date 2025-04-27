/**
 * Utility function to log errors that occur during drift analysis to a CSV file.
 * This function handles both the creation of new error log files and appending to existing ones.
 * Errors are logged with timestamps and error messages in a structured format.
 */

import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from './oneOffEmb.js';

/**
 * Creates or appends an error log entry to the error log CSV file.
 * 
 * @param {Error} error - The error object to be logged
 */
export default function makeErrorLogEntry(error) {
  // Build path to error log
  const logPath = path.join(OUTPUT_DIR, 'logs', 'ERR_log.csv');

  // Create a timestamp for when the error occurred
  const timestamp = new Date().toISOString();

  // If the file doesn't exist, this is our CSV header
  const headers = 'TIMESTAMP,MESSAGE\n';

  // Build a single error row
  // We stringify each field to handle commas, quotes, or newlines
  const row =
    [timestamp, JSON.stringify(error.message || 'Unknown error')].join(',') +
    '\n';

  // Check if the file exists already
  const fileExists = fs.existsSync(logPath);

  // If not, create the file with headers + the row
  if (!fileExists) {
    fs.writeFileSync(logPath, headers + row);
  } else {
    // Otherwise just append the new row
    fs.appendFileSync(logPath, row);
  }
}
