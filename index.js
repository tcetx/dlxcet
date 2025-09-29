#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const args = process.argv.slice(2);
if (args.length === 0) {
  console.log("Usage: dlxcet <command>");
  process.exit(0);
}

const command = args[0];

// Show experiment names
if (command === "exp") {
  const expDir = path.join(__dirname, "experiments");
  const files = fs.readdirSync(expDir).filter(f => f.endsWith(".js"));

  for (const file of files) {
    const num = file.replace(".js", "");
    const modulePath = path.join(expDir, file);
    const expModule = await import(`file://${modulePath}`);

    // Support both ESM default and CommonJS
    const exp = expModule.default || expModule;

    if (exp?.name) {
      console.log(`${num}: ${exp.name}`);
    } else {
      console.log(`${num}: [Invalid experiment file format]`);
    }
  }
}

// Experiment-specific actions
else if (!isNaN(command)) {
  const modulePath = path.join(__dirname, "experiments", `${command}.js`);
  try {
    const expModule = await import(`file://${modulePath}`);
    const exp = expModule.default;

    const flag = args[1];
    if (!flag) {
      console.log(`--- ${exp.name} ---`);
      console.log("\nTheory:\n", exp.theory);
      console.log("\nCode:\n", exp.code);
      if (exp.algorithm) console.log("\nAlgorithm:\n", exp.algorithm);
    } else if (flag === "t") {
      console.log(`--- Theory of ${exp.name} ---\n${exp.theory}`);
    } else if (flag === "c") {
      console.log(`--- Code of ${exp.name} ---\n${exp.code}`);
    } else if (flag === "a") {
      console.log(`--- Algorithm of ${exp.name} ---\n${exp.algorithm || "Not available"}`);
    } else if (flag === "v") {
      console.log(`--- Viva Questions for ${exp.name} ---`);
      if (exp.viva && exp.viva.length > 0) {
        exp.viva.forEach((item, i) => {
          console.log(`${i + 1}. Q: ${item.q}`);
          console.log(`   A: ${item.a}\n`);
        });
      } else {
        console.log("No viva questions found.");
      }
    }
  } catch (err) {
    console.log("Experiment not found.");
  }
}
else {
  console.log("Invalid command.");
}
// Hacker-style help header + simple usage (paste into index.js)
function printHelp() {
  console.log(`
██████╗  ██      ██  ██████  ████████ 
██╔══██╗ ██      ██ ██    ██    ██    
██████╔╝ ██  █   ██ ██    ██    ██    
██╔═══╝  ██ ███  ██ ██    ██    ██    
██║      ███  ████  ██████     ██    
╚═╝      ╚══   ╚══  ╚════╝     ╚═╝   

dlxcet - experiments CLI

Simple usage:
  dlxcet exp          List available experiments
  dlxcet <n>          Show theory, code, algorithm for experiment n
  dlxcet <n> t        Show only theory for experiment n
  dlxcet <n> c        Show only code for experiment n
  dlxcet <n> a        Show only algorithm for experiment n
  dlxcet <n> v        Show viva Q&A for experiment n
  dlxcet help         Show this help message
  dlxcet --help       Show this help message
`);
}

// Hook into CLI: call printHelp when user asks for help
if (command === "help" || command === "--help") {
  printHelp();
  process.exit(0);
}

