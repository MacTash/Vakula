#!/usr/bin/env node

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const environment = path.join(root, ".vakula-npm-venv");

function run(executable, args, options = {}) {
  return spawnSync(executable, args, { stdio: "inherit", ...options });
}

function findPython() {
  const candidates = [];
  if (process.env.PYTHON) candidates.push([process.env.PYTHON, []]);
  if (process.platform === "win32") {
    candidates.push(["py", ["-3"]], ["python", []]);
  } else {
    candidates.push(["python3", []], ["python", []]);
  }

  for (const [executable, prefix] of candidates) {
    const result = spawnSync(executable, [...prefix, "-c", "import sys; raise SystemExit(sys.version_info < (3, 12))"], {
      stdio: "ignore",
    });
    if (result.status === 0) return { executable, prefix };
  }
  return null;
}

function fail(message) {
  console.error(`Vakula npm install: ${message}`);
  process.exit(1);
}

const python = findPython();
if (!python) {
  fail("Python 3.12 or newer is required. Install Python, then rerun npm install.");
}

if (!fs.existsSync(environment)) {
  console.log("Vakula npm install: creating an isolated Python environment...");
  const result = run(python.executable, [...python.prefix, "-m", "venv", environment]);
  if (result.error || result.status !== 0) {
    fail("could not create the Python environment. Check that Python includes venv support.");
  }
}

const venvPython = process.platform === "win32"
  ? path.join(environment, "Scripts", "python.exe")
  : path.join(environment, "bin", "python");

const venvVersion = spawnSync(venvPython, ["-c", "import sys; raise SystemExit(sys.version_info < (3, 12))"], {
  stdio: "ignore",
});
if (venvVersion.error || venvVersion.status !== 0) {
  fail("the existing npm Python environment is older than 3.12 or damaged. Remove only .vakula-npm-venv and rerun npm install.");
}

console.log("Vakula npm install: installing Vakula and its Python dependencies...");
const result = run(venvPython, ["-m", "pip", "install", "--disable-pip-version-check", root]);
if (result.error || result.status !== 0) {
  fail("dependency installation failed. Check your network connection and rerun npm install.");
}

console.log("Vakula npm install: ready. Run `vakula` to open the terminal workspace.");
