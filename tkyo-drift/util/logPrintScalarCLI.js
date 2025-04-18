import fs from 'fs';
import path from 'path';
import chalk from 'chalk';
import Table from 'cli-table3';
import { compareScalarDistributions } from './scalarCompare.js';
import { loadScalarMetrics } from './scalarLoadMetrics.js';
import { OUTPUT_DIR } from './oneOffEmb.js';

export default async function printScalarCLI() {
  // Define the path to where scalar .jsonl files are stored
  const SCALAR_DIR = path.join(OUTPUT_DIR, 'scalars');

  // Define warning boolean to console log a warning if we are in hybrid mode
  let warn = false;
  let noRollingWarning = false;

  // Load all filenames in the scalar directory
  const files = fs.readdirSync(SCALAR_DIR);

  // Regex pattern to extract metadata from filenames:
  // Format: ioType.metric.[modelType?].baseline.scalar.jsonl
  const scalarFileRegex =
    /^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)(?:\.([a-zA-Z0-9_]+))?\.(training|rolling)\.scalar\.jsonl$/;

  // Container to group scalar file pairs (training + rolling) by metric/io/model
  const matchedPairs = new Map();

  // Step 1: Group files into training/rolling pairs by ioType + metric + modelType
  for (const file of files) {
    const match = file.match(scalarFileRegex);

    if (!match) {
      console.warn(chalk.yellow(`Skipping unrecognized file: ${file}`));
      continue;
    }

    const [_, ioType, metric, modelTypeRaw, baselineType] = match;
    const modelType = modelTypeRaw || 'shared'; // Shared metrics have no modelType
    const key = `${ioType}.${modelType}`; // Group by I/O and model type

    // Create group key if it doesn't exist
    if (!matchedPairs.has(key)) matchedPairs.set(key, {});

    // Inside that group, nest by metric
    if (!matchedPairs.get(key)[metric]) matchedPairs.get(key)[metric] = {};

    // Store file metadata
    matchedPairs.get(key)[metric][baselineType] = {
      file,
      metric,
      ioType,
      modelType,
    };
  }

  // Step 2: Print a single banner at the top of the CLI
  const banner = `SCALAR METRIC DRIFT: ROLLING vs TRAINING`;
  const pad = 12;
  const width = banner.length + pad;
  const top = '╔' + '═'.repeat(width) + '╗';
  const middle = `║${' '.repeat(pad / 2)}${banner}${' '.repeat(pad / 2)}║`;
  const bottom = '╚' + '═'.repeat(width) + '╝';
  console.log(chalk.cyanBright(`\n${top}\n${middle}\n${bottom}\n`));

  // Step 3: Loop through each (I/O + modelType) group
  for (const [groupKey, metricsObj] of matchedPairs.entries()) {
    const [ioType, modelType] = groupKey.split('.');

    // Initialize the CLI table with fixed headers
    const table = new Table({
      head: [
        chalk.bold.white('Metric'),
        chalk.bold.white('Train μ'), // Mean of training data
        chalk.bold.white('Roll μ'), // Mean of rolling data
        chalk.bold.white('Δ Mean'), // Difference in means
        chalk.bold.white('Train σ'), // Standard deviation of training data
        chalk.bold.white('Roll σ'), // Standard deviation of rolling data
        chalk.bold.white('Δ Std'), // Difference in std deviation
        chalk.bold.white('PSI'), // Population stability index
      ],
    });

    // Step 4: For each metric in this group, calculate drift values
    for (const [metric, pair] of Object.entries(metricsObj)) {
      let training;
      let rolling;

      if (!pair.rolling) {
        noRollingWarning = true;
      }
      // If we do not have a pair, we are using HYBRID MODE, and this will use both the rolling files for the training/rolling data
      if (!pair.training) {
        // Set the warning to true
        warn = true;

        training = await loadScalarMetrics(
          [metric],
          ioType,
          'rolling',
          modelType === 'shared' ? null : modelType,
          // hybrid mode is true here
          true
        );
        rolling = await loadScalarMetrics(
          [metric],
          ioType,
          'rolling',
          modelType === 'shared' ? null : modelType,
          // but not here
          false
        );
      } else {
        // If we do have a matched pair, we will use regular mode, and this will use the training and rolling files respectively.
        training = await loadScalarMetrics(
          [metric],
          ioType,
          'training',
          modelType === 'shared' ? null : modelType
          // hybrid mode is false here
        );
        rolling = await loadScalarMetrics(
          [metric],
          ioType,
          'rolling',
          modelType === 'shared' ? null : modelType
          // and also here
        );
      }

      // Compare statistical distributions (mean/std)
      const drift = compareScalarDistributions(training, rolling);
      if (!drift[metric]) {
        console.log(chalk.dim(`No data returned for ${metric}, skipping.`));
        continue;
      }

      // Push the computed values to the table
      table.push([
        metric,
        format(drift[metric].trainMean),
        format(drift[metric].rollMean),
        formatDelta(drift[metric].meanDelta, drift[metric].trainStd),
        format(drift[metric].trainStd),
        format(drift[metric].rollStd),
        formatDelta(drift[metric].stdDelta, drift[metric].trainStd),
        formatPSI(drift[metric].psi),
      ]);
    }

    // Only render tables that have valid data
    if (table.length > 0) {
      const sectionLabel = `→ ${ioType.toUpperCase()} • ${modelType.toUpperCase()} SCALAR METRIC VALUES`;
      console.log(chalk.bold.white(`\n${sectionLabel}`));
      console.log(table.toString());
    }
  }

  // Helper to color code regular values
  function format(val) {
    if (typeof val !== 'number') return chalk.gray('n/a');
    const formatted = val.toFixed(2);
    return chalk.white(formatted);
  }

  // Helper to color code delta values by severity
  function formatDelta(val, std) {
    if (typeof val !== 'number') return chalk.gray('n/a');
    const formatted = val.toFixed(2);

    const z = Math.abs(std > 0 ? val / std : 0);

    if (Math.abs(z) < 1) return chalk.green(formatted); // Safe
    if (Math.abs(z) < 2) return chalk.yellow(formatted); // Caution
    return chalk.red(formatted); // Drifted
  }

  // Helper to color code PSI values by severity
  function formatPSI(val) {
    if (typeof val !== 'number') return chalk.gray('n/a');
    const formatted = val.toFixed(3);
    if (val < 0.1) return chalk.green(formatted); // No significant change
    if (val < 0.25) return chalk.yellow(formatted); // Moderate change
    return chalk.red(formatted); // Major drift
  }

  if (warn) {
    console.log(
      chalk.gray(
        `Running in hybrid mode: Using first 10k rolling as training data. (Do you have training data?)`
      )
    );
  }

  if (noRollingWarning) {
    console.log(chalk.red(`You seem to be missing rolling data.`));
  }
}
