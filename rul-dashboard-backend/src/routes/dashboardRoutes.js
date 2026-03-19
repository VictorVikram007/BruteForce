const express = require("express");
const path = require("path");
const { execFile } = require("child_process");
const loadJson = require("../utils/loadJson");

const router = express.Router();

const getPythonCommand = () => {
  if (process.env.PYTHON_PATH) {
    return process.env.PYTHON_PATH;
  }

  if (process.platform === "win32") {
    return path.join(__dirname, "..", "..", "..", ".venv", "Scripts", "python.exe");
  }

  return "python3";
};

const getMultiModelResults = () => loadJson("data/multi-model-results.json");

router.get("/rul-prediction", (req, res) => {
  const data = loadJson("data/rul-prediction.json");
  res.status(200).json(data);
});

router.get("/degradation-trends", (req, res) => {
  const data = loadJson("data/degradation-trends.json");
  res.status(200).json(data);
});

router.get("/sensitivity-analysis", (req, res) => {
  const data = loadJson("data/sensitivity-analysis.json");
  res.status(200).json(data);
});

router.get("/multi-model-rul", (req, res) => {
  const data = getMultiModelResults();

  const models = data.models.map((model) => ({
    name: model.name,
    status: model.status,
    family: model.family || "standard",
    supports_live_inference:
      model.supports_live_inference === undefined ? true : model.supports_live_inference,
    metrics: model.metrics,
  }));

  res.status(200).json({
    dataset: data.dataset,
    rows_used: data.rows_used,
    target_column: data.target_column,
    feature_columns: data.feature_columns,
    best_model: data.best_model,
    successful_model_count: data.successful_model_count,
    comparison_summary: data.comparison_summary || null,
    models,
  });
});

router.get("/multi-model-rul/:modelName", (req, res) => {
  const data = getMultiModelResults();
  const modelName = req.params.modelName.toLowerCase();

  const model = data.models.find(
    (item) => item.name.toLowerCase() === modelName
  );

  if (!model) {
    return res.status(404).json({ message: "Requested model not found" });
  }

  return res.status(200).json({
    model: model.name,
    status: model.status,
    family: model.family || "standard",
    supports_live_inference:
      model.supports_live_inference === undefined ? true : model.supports_live_inference,
    feature_columns: model.feature_columns || data.feature_columns,
    metrics: model.metrics,
    sample_actuals: model.sample_actuals,
    sample_errors: model.sample_errors,
    sample_predictions: model.sample_predictions,
    scatter_points: model.scatter_points,
    feature_importance: model.feature_importance,
  });
});

router.get("/model-comparison", (req, res) => {
  const data = getMultiModelResults();

  const successfulModels = data.models.filter((model) => model.status === "ok");

  const payload = {
    best_model: data.best_model,
    comparison_summary: data.comparison_summary || null,
    rows_used: data.rows_used,
    target_column: data.target_column,
    feature_columns: data.feature_columns,
    models: successfulModels.map((model) => ({
      name: model.name,
      family: model.family || "standard",
      supports_live_inference:
        model.supports_live_inference === undefined ? true : model.supports_live_inference,
      metrics: model.metrics,
      scatter_points: model.scatter_points,
      sample_errors: model.sample_errors,
      feature_importance: model.feature_importance,
    })),
  };

  return res.status(200).json(payload);
});

router.post("/predict-rul", (req, res, next) => {
  const modelName = req.body?.model;
  const features = req.body?.features;

  if (!modelName || typeof modelName !== "string") {
    return res.status(400).json({ message: "model is required" });
  }

  if (!features || typeof features !== "object") {
    return res.status(400).json({ message: "features object is required" });
  }

  const scriptPath = path.join(__dirname, "..", "..", "..", "ml_models", "predict_rul.py");
  const registryPath = path.join(__dirname, "..", "..", "data", "multi-model-results.json");
  const pythonPath = getPythonCommand();

  execFile(
    pythonPath,
    [
      scriptPath,
      "--registry",
      registryPath,
      "--model",
      modelName,
      "--features-json",
      JSON.stringify(features),
    ],
    (error, stdout, stderr) => {
      if (error) {
        return next(new Error(stderr || error.message));
      }

      try {
        const parsed = JSON.parse(stdout);
        return res.status(200).json(parsed);
      } catch (parseError) {
        return next(new Error("Failed to parse prediction output"));
      }
    }
  );
});

router.post("/train-models", (req, res, next) => {
  const maxRows = Number(req.body?.maxRows) || 20000;

  const scriptPath = path.join(__dirname, "..", "..", "..", "ml_models", "train_all_models.py");
  const datasetPath = path.join(__dirname, "..", "..", "..", "battery_dataset_final.csv");
  const outputPath = path.join(
    __dirname,
    "..",
    "..",
    "data",
    "multi-model-results.json"
  );
  const pythonPath = getPythonCommand();

  execFile(
    pythonPath,
    [
      scriptPath,
      "--dataset",
      datasetPath,
      "--output",
      outputPath,
      "--max-rows",
      String(maxRows),
    ],
    (error, stdout, stderr) => {
      if (error) {
        return next(new Error(stderr || error.message));
      }

      const data = getMultiModelResults();
      return res.status(200).json({
        message: "Training complete",
        output: stdout,
        best_model: data.best_model,
        successful_model_count: data.successful_model_count,
      });
    }
  );
});

module.exports = router;