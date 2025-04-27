import path from 'path';

// TKYO Drift configuration file
//
// You can override the following settings using environment variables:
//   - TEXT_LOGGING: Set to 'false' to disable text input logging (default: true)
//   - OUTPUT_DIR: Set the output directory for all drift data (default: './tkyoData')
//
// The models object is static. To add or change models, edit this file directly.

export const config = {
  // List of transformer models to use for drift analysis. Edit this object to add/remove models.
  models: {
    mini: 'Xenova/all-MiniLM-L12-v2',
    e5: 'Xenova/e5-base-v2'
  },

  // Enable or disable logging of input text. Set TEXT_LOGGING=false in your environment to disable.
  enableTextLogging: process.env.TEXT_LOGGING === 'false' ? false : true,

  // Output directory for all drift data. Set OUTPUT_DIR in your environment to override.
  outputDir: path.resolve(process.env.OUTPUT_DIR || './tkyoData')
}; 