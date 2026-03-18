const express = require("express");
const cors = require("cors");
const path = require("path");
const dashboardRoutes = require("./routes/dashboardRoutes");

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

if (process.env.SERVE_FRONTEND === "true") {
  const staticDir = path.join(__dirname, "..", "..", "rul-dashboard-frontend", "build");
  app.use(express.static(staticDir));
}

app.get("/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.get("/api/health", (req, res) => {
  res.status(200).json({ status: "ok" });
});

app.use("/", dashboardRoutes);
app.use("/api", dashboardRoutes);

if (process.env.SERVE_FRONTEND === "true") {
  app.get("*", (req, res) => {
    const indexPath = path.join(
      __dirname,
      "..",
      "..",
      "rul-dashboard-frontend",
      "build",
      "index.html"
    );
    res.sendFile(indexPath);
  });
}

app.use((req, res) => {
  res.status(404).json({ message: "Endpoint not found" });
});

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ message: "Internal server error" });
});

app.listen(PORT, () => {
  console.log(`RUL backend server running on port ${PORT}`);
});