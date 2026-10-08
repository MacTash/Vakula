#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const environment = path.join(root, ".vakula-npm-venv");
const python = process.platform === "win32"
  ? path.join(environment, "Scripts", "python.exe")
  : path.join(environment, "bin", "python");

if (!fs.existsSync(python)) {
  console.error("Vakula's Python environment is missing. Reinstall the package with npm install scripts enabled.");
  process.exit(1);
}

const pythonPath = process.env.PYTHONPATH
  ? `${root}${path.delimiter}${process.env.PYTHONPATH}`
  : root;
const child = spawn(python, ["-m", "vakula", ...process.argv.slice(2)], {
  stdio: "inherit",
  env: {
    ...process.env,
    PYTHONPATH: pythonPath,
    PYTHONDONTWRITEBYTECODE: "1",
  },
});

child.on("error", (error) => {
  console.error(`Could not start Vakula: ${error.message}`);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  process.exitCode = signal ? 1 : (code ?? 1);
});
