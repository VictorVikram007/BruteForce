import apiClient from "./client";

export const getRulPrediction = async () => {
  const { data } = await apiClient.get("/rul-prediction");
  return data;
};

export const getDegradationTrends = async () => {
  const { data } = await apiClient.get("/degradation-trends");
  return data;
};

export const getSensitivityAnalysis = async () => {
  const { data } = await apiClient.get("/sensitivity-analysis");
  return data;
};

export const getMultiModelSummary = async () => {
  const { data } = await apiClient.get("/multi-model-rul");
  return data;
};

export const getMultiModelDetails = async (modelName) => {
  const { data } = await apiClient.get(
    `/multi-model-rul/${encodeURIComponent(modelName)}`
  );
  return data;
};

export const retrainModels = async (maxRows = 20000) => {
  const { data } = await apiClient.post("/train-models", { maxRows });
  return data;
};

export const predictRul = async (model, features) => {
  const { data } = await apiClient.post("/predict-rul", { model, features });
  return data;
};

export const getModelComparison = async () => {
  const { data } = await apiClient.get("/model-comparison");
  return data;
};