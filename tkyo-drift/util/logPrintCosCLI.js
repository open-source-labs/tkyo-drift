import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import Table from 'cli-table3';
import { MODELS, OUTPUT_DIR } from './oneOffEmb.js';

export default async function printLogCLI(arg) {
  // Constants & CLI Args
  const logPath = path.join(OUTPUT_DIR, 'logs', 'COS_log.csv');
  const days = isNaN(parseInt(arg)) ? 30 : parseInt(arg);
  const driftThreshold = 0.8;
  const startTime = Date.now() - days * 86400000; // milliseconds in a day

  // Validate that the log file actually exists
  if (!fs.existsSync(logPath)) {
    throw new Error(`No log file not found at: ${logPath}`);
  }

  // Declare header and row variables so they’re accessible later
  let headers, rows;

  try {
    // Parse the CSV file into header + rows
    const [headerLine, ...dataLines] = fs
      .readFileSync(logPath, 'utf-8')
      .trim()
      .split('\n');
    headers = headerLine.split(',');
    rows = dataLines.map((line) => line.split(','));
  } catch (error) {
    throw new Error(`Failed to parse log file: ${error.message}`);
  }

  // Dynamically find all columns that are COS values
  const COS_COLUMNS = headers.filter((h) => h.endsWith('COS'));

  // Filter rows by timestamp so we only show recent results
  const filteredRows = rows.filter((row) => {
    const timestamp = new Date(row[1]).getTime(); // index 2 = TIMESTAMP
    return timestamp >= startTime;
  });

  // I need the flippin IO Types from the column
  const ioTypes = [...new Set(filteredRows.map((row) => row[2]))];

  // Create empty maps to store cumulative similarity values
  const columnSums = {};
  const rowCounts = {};
  const baselineTypes = ['rolling', 'training'];

  // Loop through all filtered rows to aggregate cosine similarity by type
  for (const row of filteredRows) {
    const ioType = row[2]; // the I/O column

    for (const col of COS_COLUMNS) {
      const key = `${ioType}|${col}`;
      const index = headers.indexOf(col);
      const val = parseFloat(row[index]);

      if (!isNaN(val)) {
        columnSums[key] = (columnSums[key] || 0) + val;
        rowCounts[key] = (rowCounts[key] || 0) + 1;
      }
    }
  }

  // Helper: Color-code average similarity values
  const colorizeSimilarity = (val) => {
    if (val >= 0.9) return chalk.green(val.toFixed(2));
    if (val >= 0.77) return chalk.yellow(val.toFixed(2));
    return chalk.red(val.toFixed(2));
  };

  // Setup the output table
  const table = new Table({
    head: [
      chalk.bold.white('I/O Type'),
      chalk.bold.white('Drift Type'),
      chalk.bold.white('Baseline'),
      chalk.bold.white('Avg COS'),
      chalk.bold.white('Violation Count'),
    ],
  });

  // Build the table rows by model type, io type, and baseline type
  for (const ioType of ioTypes) {
    for (const [modelType] of Object.entries(MODELS)) {
      for (const baselineType of baselineTypes) {
        const columnHeader = `${modelType.toUpperCase()} ${baselineType.toUpperCase()} COS`;
        const colIndex = headers.indexOf(columnHeader);
        if (colIndex === -1) continue;

        const key = `${ioType}|${columnHeader}`;
        const sum = columnSums[key];
        const count = rowCounts[key];

        // Skip if there's no data for this combo
        if (!sum || !count) continue;

        // Clamp avg to 0–1 and prevent NaN/Infinity
        const avg = sum / count; // Math.min(1, Math.max(0, sum / count));

        // Count violations under the similarity threshold
        let groupViolations = 0;
        let groupTotal = 0;

        for (const row of filteredRows) {
          const rowIO = row[2];
          if (rowIO !== ioType) continue;

          const val = parseFloat(row[colIndex]);
          if (!isNaN(val)) {
            groupTotal++;
            if (val < driftThreshold) {
              groupViolations++;
            }
          }
        }

        // Format the violation count and severity color
        const percentValue =
          groupTotal > 0 ? Math.round((groupViolations / groupTotal) * 100) : 0;

        let coloredCount;
        if (percentValue <= 5) {
          coloredCount = chalk.green(`${groupViolations} (${percentValue}%)`);
        } else if (percentValue <= 10) {
          coloredCount = chalk.yellow(`${groupViolations} (${percentValue}%)`);
        } else {
          coloredCount = chalk.red(`${groupViolations} (${percentValue}%)`);
        }

        // Push the row to the table
        table.push([
          ioType.toUpperCase(),
          modelType.toUpperCase(),
          baselineType.toUpperCase(),
          colorizeSimilarity(avg),
          coloredCount,
        ]);
      }
    }
  }

  // Make a fancy CLI box title
  const titleText = `TKYO DRIFT ANALYTICS FOR PAST ${days} DAY(S)`;
  const padding = 24;
  const contentWidth = titleText.length + padding;
  const top = '╔' + '═'.repeat(contentWidth) + '╗';
  const middle = `║${' '.repeat(padding / 2)}${titleText}${' '.repeat(
    padding / 2
  )}║`;
  const bottom = '╚' + '═'.repeat(contentWidth) + '╝';

  // Put it all together and print
  console.log(chalk.redBright(`${top}\n${middle}\n${bottom}`));
  console.log(table.toString());
}
