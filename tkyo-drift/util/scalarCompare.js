// * Function that compares the scalar distributions between rolling and training
export function compareScalarDistributions(trainingMetrics, rollingMetrics) {
  const result = {};

  // Get the keys of metrics that exist in both data sets
  const sharedMetrics = Object.keys(trainingMetrics).filter((key) =>
    rollingMetrics.hasOwnProperty(key)
  );

  // Iterate through each shared metric key
  for (const key of sharedMetrics) {
    const train = trainingMetrics[key];
    const roll = rollingMetrics[key];

    // Skip if there is no data for that key
    if (!train.length || !roll.length) continue;

    // Get the mean and standard deviation from the training data
    const trainMean = mean(train);
    const trainStd = stddev(train);

    // Get the mean and standard deviation from the rolling data
    const rollMean = mean(roll);
    const rollStd = stddev(roll);

    result[key] = {
      trainMean,
      rollMean,
      meanDelta: rollMean - trainMean,
      trainStd,
      rollStd,
      stdDelta: rollStd - trainStd,
      psi: calculatePSI(train, roll),
    };
  }

  return result;
}

// Helper: Mean
function mean(arr) {
  return arr.reduce((sum, val) => sum + val, 0) / arr.length;
}

// Helper: Standard Deviation
function stddev(arr) {
  const avg = mean(arr);
  const variance =
    arr.reduce((sum, val) => sum + (val - avg) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

function calculatePSI(train, roll, bins = 10) {
  if (
    !Array.isArray(train) ||
    !Array.isArray(roll) ||
    train.length === 0 ||
    roll.length === 0
  ) {
    return null;
  }

  // Create common bin edges based on training data range
  const min = Math.min(...train);
  const max = Math.max(...train);
  if (min === max) return 0;

  const binEdges = [];
  for (let i = 1; i < bins; i++) {
    binEdges.push(min + ((max - min) * i) / bins);
  }

  // Helper to count frequencies per bin
  function getBinFreqs(values) {
    const freqs = new Array(bins).fill(0);
    for (const val of values) {
      const idx = binEdges.findIndex((edge) => val < edge);
      freqs[idx === -1 ? bins - 1 : idx]++;
    }
    return freqs.map((f) => f / values.length);
  }

  const trainFreq = getBinFreqs(train);
  const rollFreq = getBinFreqs(roll);

  // Calculate PSI: sum of (train % - roll %) * ln(train % / roll %)
  let psi = 0;
  for (let i = 0; i < bins; i++) {
    const t = trainFreq[i] || 1e-6;
    const r = rollFreq[i] || 1e-6;
    psi += (t - r) * Math.log(t / r);
  }

  return psi;
}
