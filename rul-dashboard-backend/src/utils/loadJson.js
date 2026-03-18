const fs = require("fs");
const path = require("path");

const loadJson = (relativePath) => {
  const absolutePath = path.join(__dirname, "..", "..", relativePath);
  const file = fs.readFileSync(absolutePath, "utf8");
  return JSON.parse(file);
};

module.exports = loadJson;