/**
 * Utility function to log text inputs that are being analyzed for drift detection.
 * This creates and maintains a CSV log file that tracks the text inputs along with their
 * unique identifiers and timestamps. This log is separate from the drift metrics log
 * to keep the input data distinct from the analysis results.
 */

import fs from 'fs';
import path from 'path';
import { OUTPUT_DIR } from './oneOffEmb.js';

/**
 * Escapes a string for CSV format by:
 * 1. Replacing any double quotes with two double quotes
 * 2. Wrapping the entire string in double quotes
 * 
 * @param {string} str - The string to escape
 * @returns {string} - The escaped string
 */
function escapeCSV(str) {
  // Replace any double quotes with two double quotes
  const escaped = str.replace(/"/g, '""');
  // Wrap in double quotes
  return `"${escaped}"`;
}

/**
 * Creates or appends a log entry for a text input being analyzed for drift.
 * 
 * @param {string} id - Unique identifier (UUID) for the text input
 * @param {string} text - The actual text content being analyzed
 */
export default function logMakeInputEntry(id, text) {
  // Construct the path to the text log file in the logs directory
  const logPath = path.join(OUTPUT_DIR, 'logs', 'text_log.csv');
  
  // Generate an ISO timestamp for when this entry is being made
  const timestamp = new Date().toISOString();
  
  // Create the CSV row with ID, timestamp, and escaped text content
  // ID and timestamp don't need escaping as they won't contain special characters
  const row = [id, timestamp, escapeCSV(text)];
  const csvLine = row.join(',') + '\n';

  // Check if the log file already exists
  const fileExists = fs.existsSync(logPath);

  // Write to the log file
  try {
    if (!fileExists) {
      // If the file doesn't exist, create it with headers
      const headers = ['ID', 'TIMESTAMP', 'TEXT'].join(',') + '\n';
      fs.writeFileSync(logPath, headers + csvLine);
    } else {
      // If the file exists, append the new entry
      fs.appendFileSync(logPath, csvLine);
    }
  } catch (error) {
    // Log any errors that occur during file operations
    // This could be due to permissions, disk space, or file locks
    console.error('Failed to write text log entry:', error.message);
  }
} 