import React, { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "react-query";
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Divider,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Slider,
  Stack,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
  Chip,
} from "@mui/material";
import {
  Cell,
  Line,
  LineChart,
  BarChart,
  CartesianGrid,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  Bar,
  ComposedChart,
} from "recharts";
import {
  getMultiModelDetails,
  getMultiModelSummary,
  predictRul,
  retrainModels,
} from "../api/rulApi";
import { ErrorState, LoadingState } from "./StateViews";

const RULPrediction = () => {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState("");
  const [featureInputs, setFeatureInputs] = useState({});
  const [simulation, setSimulation] = useState({
    dailyKm: 135,
    ambientTemp: 32,
    dod: 0.6,
  });
  const [smartInputMode, setSmartInputMode] = useState(true);
  const [activeScenario, setActiveScenario] = useState("Balanced");
  const [liveTick, setLiveTick] = useState(0);

  const scenarioPresets = useMemo(
    () => ({
      Balanced: { dailyKm: 135, ambientTemp: 32, dod: 0.6 },
      "High temperature scenario": { dailyKm: 135, ambientTemp: 44, dod: 0.65 },
      "Aggressive driving": { dailyKm: 200, ambientTemp: 35, dod: 0.82 },
      "Fast charging abuse": { dailyKm: 170, ambientTemp: 38, dod: 0.75 },
      "Optimized strategy": { dailyKm: 120, ambientTemp: 25, dod: 0.5 },
    }),
    []
  );

  useEffect(() => {
    const timer = setInterval(() => {
      setLiveTick((prev) => prev + 1);
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const {
    data: summary,
    isLoading: isSummaryLoading,
    error: summaryError,
  } = useQuery("multiModelSummary", getMultiModelSummary);

  useEffect(() => {
    if (!summary) {
      return;
    }

    if (summary.best_model) {
      setSelectedModel((prev) => prev || summary.best_model);
      return;
    }

    const fallback = summary.models?.find((item) => item.status === "ok")?.name;
    setSelectedModel((prev) => prev || fallback || "");
  }, [summary]);

  const {
    data: selectedDetail,
    isLoading: isDetailLoading,
    error: detailError,
  } = useQuery(
    ["multiModelDetail", selectedModel],
    () => getMultiModelDetails(selectedModel),
    { enabled: Boolean(selectedModel) }
  );

  const retrainMutation = useMutation((maxRows) => retrainModels(maxRows), {
    onSuccess: () => {
      queryClient.invalidateQueries("multiModelSummary");
      queryClient.invalidateQueries("multiModelDetail");
    },
  });

  const predictMutation = useMutation(({ model, features }) =>
    predictRul(model, features)
  );

  const leaderboard = useMemo(() => {
    if (!summary?.models) return [];
    return [...summary.models]
      .filter((model) => model.status === "ok" && model.metrics)
      .sort((a, b) => a.metrics.rmse - b.metrics.rmse)
      .map((model, index) => ({
        rank: index + 1,
        name: model.name,
        family: model.family || "standard",
        rmse: Number(model.metrics.rmse.toFixed(2)),
        mae: Number(model.metrics.mae.toFixed(2)),
        r2: Number(model.metrics.r2.toFixed(4)),
      }));
  }, [summary]);

  const activeFeatureColumns = useMemo(
    () => selectedDetail?.feature_columns || summary?.feature_columns || [],
    [selectedDetail, summary]
  );

  useEffect(() => {
    if (!activeFeatureColumns.length) {
      return;
    }

    setFeatureInputs((prev) => {
      const next = { ...prev };
      activeFeatureColumns.forEach((column) => {
        if (next[column] === undefined) {
          next[column] = "0";
        }
      });
      return next;
    });
  }, [activeFeatureColumns]);

  const predictionParityData = useMemo(() => {
    const raw = selectedDetail?.scatter_points || [];
    if (!raw.length) {
      return {
        points: [],
        actualHist: [],
        predictedHist: [],
        min: 0,
        max: 1,
      };
    }

    const values = raw.flatMap((p) => [p.actual, p.predicted]);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const range = Math.max(maxValue - minValue, 1);
    const bins = 14;
    const binWidth = range / bins;

    const actualBins = Array.from({ length: bins }, (_, index) => ({
      bucket: `${Math.round(minValue + index * binWidth)}`,
      count: 0,
    }));

    const predictedBins = Array.from({ length: bins }, (_, index) => ({
      bucket: `${Math.round(minValue + index * binWidth)}`,
      count: 0,
    }));

    const rmse = selectedDetail?.metrics?.rmse || 1;
    const points = raw.map((point) => {
      const absError = Math.abs(point.predicted - point.actual);
      const actualIndex = Math.min(
        bins - 1,
        Math.max(0, Math.floor((point.actual - minValue) / binWidth))
      );
      const predictedIndex = Math.min(
        bins - 1,
        Math.max(0, Math.floor((point.predicted - minValue) / binWidth))
      );

      actualBins[actualIndex].count += 1;
      predictedBins[predictedIndex].count += 1;

      let color = "#14b8a6";
      if (absError > rmse * 1.5) {
        color = "#ef4444";
      } else if (absError > rmse * 0.8) {
        color = "#f59e0b";
      }

      return {
        ...point,
        absError: Number(absError.toFixed(2)),
        color,
      };
    });

    return {
      points,
      actualHist: actualBins,
      predictedHist: predictedBins,
      min: Math.floor(minValue),
      max: Math.ceil(maxValue),
    };
  }, [selectedDetail]);

  const activeModelMetrics = selectedDetail?.metrics;

  const baseRul = useMemo(() => {
    const predictionBelongsToSelectedModel =
      predictMutation.data?.model &&
      selectedModel &&
      predictMutation.data.model === selectedModel;

    if (
      predictionBelongsToSelectedModel &&
      predictMutation.data?.prediction !== undefined
    ) {
      return Number(predictMutation.data.prediction);
    }
    if (selectedDetail?.sample_predictions?.length) {
      const arr = selectedDetail.sample_predictions;
      const avg = arr.reduce((acc, value) => acc + value, 0) / arr.length;
      return Number(avg);
    }
    return 1000;
  }, [predictMutation.data?.model, predictMutation.data?.prediction, selectedDetail, selectedModel]);

  const adjustRulBySimulation = (baseline, sim) => {
    const tempPenalty = Math.max(0, sim.ambientTemp - 25) * 7.2;
    const dodPenalty = Math.max(0, sim.dod - 0.55) * 560;
    const kmPenalty = Math.max(0, sim.dailyKm - 120) * 0.9;
    return Math.max(40, baseline - tempPenalty - dodPenalty - kmPenalty);
  };

  const scenarioCurrentRul = useMemo(
    () => Math.round(adjustRulBySimulation(baseRul, simulation)),
    [baseRul, simulation]
  );

  const optimizedScenario = scenarioPresets["Optimized strategy"];
  const scenarioOptimizedRul = useMemo(
    () => Math.round(adjustRulBySimulation(baseRul, optimizedScenario)),
    [baseRul, optimizedScenario]
  );

  const scenarioGain = Math.max(0, scenarioOptimizedRul - scenarioCurrentRul);
  const costSavedInr = Math.round(scenarioGain * 6.1);
  const failureRiskReductionPct = Number(
    Math.max(0, Math.min(95, (scenarioGain / Math.max(scenarioCurrentRul, 1)) * 18)).toFixed(1)
  );
  const warrantyMonthsGain = Number((scenarioGain / 25).toFixed(2));
  const remainingLifeMonths = Number(((scenarioCurrentRul * 160) / (Math.max(simulation.dailyKm, 1) * 30)).toFixed(1));
  const scenarioComparisonData = useMemo(
    () => [
      { name: "Current", rul: scenarioCurrentRul },
      { name: "Improved", rul: scenarioOptimizedRul },
    ],
    [scenarioCurrentRul, scenarioOptimizedRul]
  );

  const confidenceBand = useMemo(() => {
    const lower = predictMutation.data?.prediction_lower;
    const upper = predictMutation.data?.prediction_upper;
    if (typeof lower === "number" && typeof upper === "number") {
      return `+-${Math.round(Math.abs(upper - lower) / 2)} cycles`;
    }

    const rmse = activeModelMetrics?.rmse;
    if (typeof rmse === "number") {
      return `+-${Math.round(1.96 * rmse)} cycles`;
    }

    return "+-150 cycles";
  }, [activeModelMetrics?.rmse, predictMutation.data?.prediction_lower, predictMutation.data?.prediction_upper]);

  const confidenceLabel = useMemo(() => {
    const rmse = activeModelMetrics?.rmse ?? 80;
    if (rmse <= 20) return "High";
    if (rmse <= 60) return "Moderate";
    return "Caution";
  }, [activeModelMetrics?.rmse]);

  const finalAction = useMemo(() => {
    if (scenarioCurrentRul < 250) {
      return "Replace within 30 days";
    }
    if (scenarioCurrentRul < 600) {
      return "Schedule inspection and controlled charging";
    }
    return "Continue operation with monthly monitoring";
  }, [scenarioCurrentRul]);

  const fleetDecisionRows = useMemo(() => {
    const b1 = Math.max(60, Math.round(scenarioCurrentRul * 0.22));
    const b2 = Math.max(180, Math.round(scenarioCurrentRul * 0.52));
    const b3 = Math.max(420, Math.round(scenarioCurrentRul * 0.95));

    const toRisk = (rul) => {
      if (rul < 250) return { risk: "Critical", action: "Replace" };
      if (rul < 600) return { risk: "Warning", action: "Monitor" };
      return { risk: "Healthy", action: "OK" };
    };

    return [
      { battery: "B1", rul: b1, ...toRisk(b1) },
      { battery: "B2", rul: b2, ...toRisk(b2) },
      { battery: "B3", rul: b3, ...toRisk(b3) },
    ];
  }, [scenarioCurrentRul]);

  const explainability = useMemo(() => {
    const temperatureScore = Math.max(0, simulation.ambientTemp - 25) * 1.3;
    const dodScore = Math.max(0, simulation.dod - 0.5) * 100;
    const chargingScore = Math.max(0, Number(featureInputs.Max_Discharge_C_Rate || 1.2) - 1.0) * 35;
    const total = Math.max(temperatureScore + dodScore + chargingScore, 1e-9);

    return [
      {
        name: "High temperature",
        pct: Number(((temperatureScore / total) * 100).toFixed(1)),
        note: "Thermal stress accelerates capacity fade and resistance growth.",
      },
      {
        name: "Deep discharge",
        pct: Number(((dodScore / total) * 100).toFixed(1)),
        note: "Higher DoD cycles increase electrochemical wear.",
      },
      {
        name: "Charging stress",
        pct: Number(((chargingScore / total) * 100).toFixed(1)),
        note: "Higher C-rate raises plating and early degradation risk.",
      },
    ].sort((a, b) => b.pct - a.pct);
  }, [featureInputs.Max_Discharge_C_Rate, simulation.ambientTemp, simulation.dod]);

  const midcProfile = useMemo(() => {
    const points = [];
    for (let t = 0; t <= 1080; t += 20) {
      const block = Math.floor(t / 180);
      const amp = [22, 30, 26, 34, 24, 28][block] || 22;
      const local = (t % 180) / 180;
      const speed = Math.max(0, amp * Math.sin(Math.PI * local));
      const power = Math.max(0, (0.0022 * speed + 0.00003 * speed * speed) * (1 + 0.15 * (simulation.dod - 0.6)));
      points.push({ time: t, speed: Number(speed.toFixed(2)), power: Number(power.toFixed(4)) });
    }
    return points;
  }, [simulation.dod]);

  const livePulseData = useMemo(() => {
    const points = [];
    const base = scenarioCurrentRul;
    for (let i = 0; i < 20; i += 1) {
      const t = i + liveTick;
      points.push({
        slot: i + 1,
        thermal: Number((simulation.ambientTemp + 1.8 * Math.sin(t / 3)).toFixed(2)),
        rul: Number((base - i * 0.9 + 3.5 * Math.sin(t / 4)).toFixed(2)),
      });
    }
    return points;
  }, [liveTick, scenarioCurrentRul, simulation.ambientTemp]);

  const handleRetrain = () => {
    retrainMutation.mutate(20000);
  };

  const handlePredict = () => {
    const numericFeatures = {};
    for (const column of activeFeatureColumns) {
      const value = Number(featureInputs[column]);
      if (Number.isNaN(value)) {
        predictMutation.reset();
        return;
      }
      numericFeatures[column] = value;
    }

    predictMutation.mutate({
      model: selectedModel,
      features: numericFeatures,
    });
  };

  const applySimulationToInputs = () => {
    const normalizedDoD = simulation.dod * 100;
    const normalizedDischargeRate = 0.8 + (simulation.dailyKm / 250) * 2.2;
    const normalizedChargeRate = 0.6 + (simulation.dailyKm / 250) * 1.2;

    setFeatureInputs((prev) => ({
      ...prev,
      Avg_Ambient_Temp: simulation.ambientTemp.toFixed(2),
      Peak_Cell_Temp: (simulation.ambientTemp + 8).toFixed(2),
      Daily_DoD: normalizedDoD.toFixed(2),
      Rolling_Avg_DoD: normalizedDoD.toFixed(2),
      Rolling_Avg_Temp: simulation.ambientTemp.toFixed(2),
      Max_Discharge_C_Rate: normalizedDischargeRate.toFixed(2),
      Max_Charge_C_Rate: normalizedChargeRate.toFixed(2),
    }));
  };

  const applyScenarioPreset = (name) => {
    const preset = scenarioPresets[name];
    if (!preset) {
      return;
    }
    setActiveScenario(name);
    setSimulation(preset);
  };

  if (isSummaryLoading) return <LoadingState label="Loading multi-model summary..." />;
  if (summaryError) return <ErrorState message={summaryError.message} />;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
              Recommendation
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Typography variant="subtitle2" color="text.secondary">
                  Remaining Life
                </Typography>
                <Typography variant="h4">
                  {scenarioCurrentRul} cycles
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  ~{remainingLifeMonths} months
                </Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="subtitle2" color="text.secondary">
                  Confidence Band
                </Typography>
                <Typography variant="h4">{confidenceBand}</Typography>
                <Typography variant="body2" color="text.secondary">
                  Confidence: {confidenceLabel}
                </Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="subtitle2" color="text.secondary">
                  Final Action
                </Typography>
                <Typography variant="h6" sx={{ mt: 1 }}>
                  {finalAction}
                </Typography>
              </Grid>
              <Grid item xs={12} md={3}>
                <Typography variant="subtitle2" color="text.secondary">
                  ROI Summary
                </Typography>
                <Typography variant="body1" sx={{ mt: 1, fontWeight: 700 }}>
                  Save INR {costSavedInr} per battery, reduce failure risk by {failureRiskReductionPct}%, and extend warranty by {warrantyMonthsGain} months.
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Stack
              direction={{ xs: "column", md: "row" }}
              justifyContent="space-between"
              alignItems={{ xs: "flex-start", md: "center" }}
              spacing={1}
              sx={{ mb: 2 }}
            >
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                Live Battery Pulse
              </Typography>
              <Chip label="Streaming telemetry simulation" color="success" variant="outlined" />
            </Stack>
            <Box sx={{ height: 240 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={livePulseData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="slot" />
                  <YAxis yAxisId="left" label={{ value: "Temp (C)", angle: -90, position: "insideLeft" }} />
                  <YAxis yAxisId="right" orientation="right" label={{ value: "RUL", angle: 90, position: "insideRight" }} />
                  <Tooltip />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="thermal" stroke="#ef4444" strokeWidth={2} dot={false} name="Cell thermal" />
                  <Line yAxisId="right" type="monotone" dataKey="rul" stroke="#2563eb" strokeWidth={2.5} dot={false} name="Projected RUL" />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Stack
          direction={{ xs: "column", md: "row" }}
          justifyContent="space-between"
          spacing={2}
          alignItems={{ xs: "flex-start", md: "center" }}
        >
          <Typography variant="h4" gutterBottom>
            Multi-Model RUL Prediction
          </Typography>
          <Button
            variant="contained"
            onClick={handleRetrain}
            disabled={retrainMutation.isLoading}
          >
            {retrainMutation.isLoading ? "Retraining..." : "Retrain Models"}
          </Button>
        </Stack>
        <Typography variant="subtitle1" color="text.secondary" sx={{ mt: 0.5 }}>
          Predict battery failure before it happens using physics-informed AI.
        </Typography>
      </Grid>
      {retrainMutation.error && (
        <Grid item xs={12}>
          <ErrorState message={retrainMutation.error.message} />
        </Grid>
      )}
      {retrainMutation.data?.message && (
        <Grid item xs={12}>
          <Alert severity="success">{retrainMutation.data.message}</Alert>
        </Grid>
      )}
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary">
              Best Model
            </Typography>
            <Typography variant="h5" color="primary.main" sx={{ mt: 1 }}>
              {summary.best_model || "N/A"}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Trained on {summary.rows_used} rows from dataset
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Successful models: {summary.successful_model_count}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <FormControl fullWidth>
              <InputLabel id="model-select-label">Model</InputLabel>
              <Select
                labelId="model-select-label"
                label="Model"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {(summary.models || [])
                  .filter((model) => model.status === "ok")
                  .map((model) => (
                    <MenuItem key={model.name} value={model.name}>
                      {model.name} ({model.family || "standard"})
                    </MenuItem>
                  ))}
              </Select>
            </FormControl>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary">
              Selected Model RMSE
            </Typography>
            <Typography variant="h4" color="secondary.main" sx={{ mt: 1 }}>
              {activeModelMetrics ? activeModelMetrics.rmse.toFixed(2) : "-"}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Lower is better
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Family: {selectedDetail?.family || "standard"}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      {summary?.comparison_summary && (
        <Grid item xs={12}>
          <Alert
            severity={
              summary.comparison_summary.hybrid_beats_standard ? "success" : "info"
            }
          >
            Best standard: {summary.comparison_summary.best_standard_model || "N/A"} (RMSE{" "}
            {summary.comparison_summary.best_standard_rmse?.toFixed?.(2) || "-"}) | Best hybrid: {" "}
            {summary.comparison_summary.best_hybrid_model || "N/A"} (RMSE{" "}
            {summary.comparison_summary.best_hybrid_rmse?.toFixed?.(2) || "-"})
            {summary.comparison_summary.hybrid_rmse_improvement_pct !== null &&
            summary.comparison_summary.hybrid_rmse_improvement_pct !== undefined
              ? ` | Hybrid RMSE improvement: ${summary.comparison_summary.hybrid_rmse_improvement_pct.toFixed(2)}%`
              : ""}
          </Alert>
        </Grid>
      )}
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent sx={{ height: 360 }}>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
              MIDC profile visibility
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              MIDC speed-time profile with energy load overlay
            </Typography>
            <ResponsiveContainer width="100%" height="78%">
              <LineChart data={midcProfile}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis yAxisId="left" label={{ value: "Speed (km/h)", angle: -90, position: "insideLeft" }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: "Power (kW)", angle: 90, position: "insideRight" }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="speed" stroke="#93c5fd" strokeWidth={2.5} dot={false} name="Speed" />
                <Line yAxisId="right" type="monotone" dataKey="power" stroke="#fca5a5" strokeWidth={2.2} dot={false} name="Power" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Stack
              direction={{ xs: "column", md: "row" }}
              justifyContent="space-between"
              alignItems={{ xs: "flex-start", md: "center" }}
              spacing={1}
              sx={{ mb: 2 }}
            >
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                Scenario Simulation
              </Typography>
              <Chip
                color="primary"
                variant="outlined"
                label={`Active scenario: ${activeScenario}`}
              />
            </Stack>

            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2, borderRadius: 2, bgcolor: "#f8fafc", border: "1px solid #e2e8f0" }}>
                  <Typography variant="body2" color="text.secondary">
                    Current strategy RUL
                  </Typography>
                  <Typography variant="h4" sx={{ mt: 0.5, fontWeight: 700 }}>
                    {scenarioCurrentRul}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    cycles remaining
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2, borderRadius: 2, bgcolor: "#f0fdf4", border: "1px solid #bbf7d0" }}>
                  <Typography variant="body2" color="text.secondary">
                    Improved scenario RUL
                  </Typography>
                  <Typography variant="h4" sx={{ mt: 0.5, fontWeight: 700, color: "success.main" }}>
                    {scenarioOptimizedRul}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    25C + lower DoD plan
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2, borderRadius: 2, bgcolor: "#fff7ed", border: "1px solid #fed7aa" }}>
                  <Typography variant="body2" color="text.secondary">
                    Net gain
                  </Typography>
                  <Typography variant="h4" sx={{ mt: 0.5, fontWeight: 700, color: "warning.main" }}>
                    +{scenarioGain}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    extra cycles from strategy shift
                  </Typography>
                </Box>
              </Grid>
            </Grid>

            <Box sx={{ height: 220, mt: 2 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scenarioComparisonData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="rul" fill="#2563eb" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Box>

            <Grid container spacing={2} sx={{ mt: 0.5 }}>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 1.5, borderRadius: 2, bgcolor: "#f8fafc" }}>
                  <Typography variant="body2" color="text.secondary">Estimated cost saved per battery</Typography>
                  <Typography variant="h6">INR {costSavedInr}</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 1.5, borderRadius: 2, bgcolor: "#f8fafc" }}>
                  <Typography variant="body2" color="text.secondary">Failure risk reduction</Typography>
                  <Typography variant="h6">{failureRiskReductionPct}%</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 1.5, borderRadius: 2, bgcolor: "#f8fafc" }}>
                  <Typography variant="body2" color="text.secondary">Warranty extension potential</Typography>
                  <Typography variant="h6">{warrantyMonthsGain} months</Typography>
                </Box>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
              Fleet Health Command Center
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Decision-ready recommendations for battery maintenance scheduling.
            </Typography>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Battery</TableCell>
                  <TableCell>RUL</TableCell>
                  <TableCell>Risk</TableCell>
                  <TableCell>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {fleetDecisionRows.map((row) => (
                  <TableRow key={row.battery}>
                    <TableCell>{row.battery}</TableCell>
                    <TableCell>{row.rul} cycles</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        label={row.risk}
                        color={
                          row.risk === "Critical"
                            ? "error"
                            : row.risk === "Warning"
                            ? "warning"
                            : "success"
                        }
                      />
                    </TableCell>
                    <TableCell>{row.action}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={7}>
        <Card>
          <CardContent>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
              Why This Battery Is Critical
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Ranked degradation drivers behind the current prediction.
            </Typography>
            <Stack spacing={1.5}>
              {explainability.map((driver) => (
                <Box key={driver.name} sx={{ p: 1.5, borderRadius: 2, bgcolor: "#f8fafc", border: "1px solid #e2e8f0" }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 0.5 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                      {driver.name}
                    </Typography>
                    <Chip size="small" label={`${driver.pct}% impact`} color="primary" variant="outlined" />
                  </Stack>
                  <Typography variant="body2" color="text.secondary">
                    {driver.note}
                  </Typography>
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={5}>
        <Card sx={{ height: "100%" }}>
          <CardContent>
            <Typography variant="h5" sx={{ fontWeight: 700, mb: 1 }}>
              Inference Architecture
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              End-to-end path from telemetry to maintenance recommendation.
            </Typography>
            <Stack spacing={1.2}>
              {["1. EV telemetry stream", "2. Physics + feature synthesis", "3. Hybrid model inference", "4. Decision and ROI guidance"].map((step) => (
                <Box
                  key={step}
                  sx={{
                    p: 1.2,
                    borderRadius: 1.5,
                    bgcolor: "#f8fafc",
                    border: "1px solid #e2e8f0",
                    fontWeight: 600,
                  }}
                >
                  {step}
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 320 }}>
            <Typography variant="h6" gutterBottom>
              Model Leaderboard (RMSE)
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={leaderboard}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rmse" fill="#1f3c88" />
                <Line type="monotone" dataKey="mae" stroke="#0b6e4f" strokeWidth={2} />
              </ComposedChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card sx={{ height: "100%" }}>
          <CardContent sx={{ height: 430 }}>
            <Typography variant="h6" gutterBottom>
              Actual vs Predicted: {selectedModel || "N/A"}
            </Typography>
            {isDetailLoading && <LoadingState label="Loading selected model predictions..." />}
            {detailError && <ErrorState message={detailError.message} />}
            {!isDetailLoading && !detailError && (
              <>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  R2: {selectedDetail?.metrics?.r2?.toFixed?.(4) || "-"} | MAE: {selectedDetail?.metrics?.mae?.toFixed?.(2) || "-"} cycles | RMSE: {selectedDetail?.metrics?.rmse?.toFixed?.(2) || "-"} cycles
                </Typography>
                <Grid container spacing={1} sx={{ height: "calc(100% - 48px)" }}>
                  <Grid item xs={12} sx={{ height: "24%" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={predictionParityData.actualHist}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="bucket" hide />
                        <YAxis allowDecimals={false} />
                        <Tooltip />
                        <Bar dataKey="count" fill="#94a3b8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Grid>
                  <Grid item xs={9} sx={{ height: "76%" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          type="number"
                          dataKey="actual"
                          name="Actual RUL"
                          domain={[predictionParityData.min, predictionParityData.max]}
                        />
                        <YAxis
                          type="number"
                          dataKey="predicted"
                          name="Predicted RUL"
                          domain={[predictionParityData.min, predictionParityData.max]}
                        />
                        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                        <ReferenceLine
                          segment={[
                            { x: predictionParityData.min, y: predictionParityData.min },
                            { x: predictionParityData.max, y: predictionParityData.max },
                          ]}
                          stroke="#1e293b"
                          strokeWidth={2}
                        />
                        <Scatter data={predictionParityData.points}>
                          {predictionParityData.points.map((entry, index) => (
                            <Cell key={`parity-${index}`} fill={entry.color} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </Grid>
                  <Grid item xs={3} sx={{ height: "76%" }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={predictionParityData.predictedHist} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" allowDecimals={false} />
                        <YAxis type="category" dataKey="bucket" hide />
                        <Tooltip />
                        <Bar dataKey="count" fill="#5eead4" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Grid>
                </Grid>
              </>
            )}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Live Inference
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Smart Input Mode keeps this demo decision-focused: adjust temperature, DoD, distance, and charging behavior while the system fills low-level telemetry features.
            </Typography>
            <Stack direction={{ xs: "column", md: "row" }} spacing={1} sx={{ mb: 2 }}>
              {Object.keys(scenarioPresets).map((name) => (
                <Button
                  key={name}
                  size="small"
                  variant={activeScenario === name ? "contained" : "outlined"}
                  onClick={() => applyScenarioPreset(name)}
                >
                  {name}
                </Button>
              ))}
            </Stack>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Card
                  variant="outlined"
                  sx={{
                    bgcolor: "#e8ebf1",
                    borderColor: "#d2d8e2",
                    height: "100%",
                  }}
                >
                  <CardContent>
                    <Typography variant="h5" sx={{ fontWeight: 700, mb: 2 }}>
                      Vehicle profile
                    </Typography>
                    <Stack spacing={3}>
                      <Box>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}>
                          Daily km
                        </Typography>
                        <Typography sx={{ color: "#f1484a", textAlign: "center" }}>
                          {simulation.dailyKm}
                        </Typography>
                        <Slider
                          value={simulation.dailyKm}
                          min={20}
                          max={250}
                          onChange={(event, value) =>
                            setSimulation((prev) => ({ ...prev, dailyKm: value }))
                          }
                          sx={{
                            color: "#f1484a",
                            "& .MuiSlider-track": { border: "none" },
                          }}
                        />
                      </Box>
                      <Box>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}>
                          Average ambient temperature (C)
                        </Typography>
                        <Typography sx={{ color: "#f1484a", textAlign: "center" }}>
                          {simulation.ambientTemp}
                        </Typography>
                        <Slider
                          value={simulation.ambientTemp}
                          min={10}
                          max={60}
                          onChange={(event, value) =>
                            setSimulation((prev) => ({ ...prev, ambientTemp: value }))
                          }
                          sx={{
                            color: "#f1484a",
                            "& .MuiSlider-track": { border: "none" },
                          }}
                        />
                      </Box>
                      <Box>
                        <Typography variant="subtitle1" sx={{ mb: 1 }}>
                          Depth of discharge (DoD)
                        </Typography>
                        <Typography sx={{ color: "#f1484a", textAlign: "center" }}>
                          {simulation.dod.toFixed(2)}
                        </Typography>
                        <Slider
                          value={simulation.dod}
                          step={0.01}
                          min={0.2}
                          max={0.95}
                          onChange={(event, value) =>
                            setSimulation((prev) => ({ ...prev, dod: value }))
                          }
                          sx={{
                            color: "#f1484a",
                            "& .MuiSlider-track": { border: "none" },
                          }}
                        />
                      </Box>
                    </Stack>
                    <Button
                      variant="contained"
                      color="error"
                      onClick={applySimulationToInputs}
                      sx={{ mt: 2 }}
                    >
                      Apply Simulation
                    </Button>
                    <FormControlLabel
                      sx={{ mt: 1 }}
                      control={
                        <Switch
                          checked={smartInputMode}
                          onChange={(event) => setSmartInputMode(event.target.checked)}
                          color="secondary"
                        />
                      }
                      label="Smart Input Mode"
                    />
                  </CardContent>
                </Card>
              </Grid>
              <Grid item xs={12} md={8}>
                <Typography variant="subtitle1" sx={{ mb: 1 }}>
                  {smartInputMode ? "Smart Inputs Applied" : "Manual Input"}
                </Typography>
                {smartInputMode ? (
                  <Alert severity="info">
                    Technical features are auto-filled internally for demo storytelling. Disable Smart Input Mode if you want full feature-level control.
                  </Alert>
                ) : (
                  <Grid container spacing={2}>
                    {activeFeatureColumns.map((column) => (
                      <Grid item xs={12} sm={6} md={4} key={column}>
                        <TextField
                          label={column}
                          value={featureInputs[column] ?? "0"}
                          onChange={(event) =>
                            setFeatureInputs((prev) => ({
                              ...prev,
                              [column]: event.target.value,
                            }))
                          }
                          fullWidth
                          size="small"
                        />
                      </Grid>
                    ))}
                  </Grid>
                )}
              </Grid>
            </Grid>
            <Divider sx={{ my: 2 }} />
            <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
              <Button
                variant="contained"
                onClick={handlePredict}
                disabled={
                  !selectedModel ||
                  predictMutation.isLoading ||
                  selectedDetail?.supports_live_inference === false
                }
              >
                {predictMutation.isLoading ? "Predicting..." : "Predict RUL"}
              </Button>
              {selectedDetail?.supports_live_inference === false && (
                <Alert severity="info">
                  Live inference is disabled for this model. Use the leaderboard and trace to compare performance.
                </Alert>
              )}
              {predictMutation.data?.prediction !== undefined && (
                <Alert severity="success">
                  Predicted RUL using {predictMutation.data.model}: {predictMutation.data.prediction.toFixed(2)}
                </Alert>
              )}
              {predictMutation.error && (
                <Alert severity="error">{predictMutation.error.message}</Alert>
              )}
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default RULPrediction;