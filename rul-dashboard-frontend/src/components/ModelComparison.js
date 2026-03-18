import React, { useMemo, useState } from "react";
import { useQuery } from "react-query";
import {
  Card,
  CardContent,
  FormControl,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  Typography,
} from "@mui/material";
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getModelComparison } from "../api/rulApi";
import { ErrorState, LoadingState } from "./StateViews";

const ModelComparison = () => {
  const { data, isLoading, error } = useQuery("modelComparison", getModelComparison);
  const [selectedModel, setSelectedModel] = useState("");

  const modelNames = useMemo(
    () => (data?.models || []).map((model) => model.name),
    [data]
  );

  const activeModelName = selectedModel || data?.best_model || modelNames[0] || "";

  const activeModel = useMemo(
    () => (data?.models || []).find((model) => model.name === activeModelName),
    [activeModelName, data]
  );

  const leaderboard = useMemo(() => {
    if (!data?.models) {
      return [];
    }

    return [...data.models]
      .sort((a, b) => a.metrics.rmse - b.metrics.rmse)
      .map((model) => ({
        name: `${model.name} (${model.family || "standard"})`,
        rmse: Number(model.metrics.rmse.toFixed(2)),
        mae: Number(model.metrics.mae.toFixed(2)),
      }));
  }, [data]);

  const importanceSeries = useMemo(
    () =>
      (activeModel?.feature_importance || []).slice(0, 10).map((item) => ({
        feature: item.feature,
        importance: Number(item.importance.toFixed(4)),
      })),
    [activeModel]
  );

  const parityData = useMemo(() => {
    const raw = activeModel?.scatter_points || [];
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

    const rmse = activeModel?.metrics?.rmse || 1;
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
  }, [activeModel]);

  const errorHistogram = useMemo(() => {
    const errors = activeModel?.sample_errors || [];
    if (!errors.length) {
      return [];
    }

    const maxErr = Math.max(...errors);
    const bins = 12;
    const width = Math.max(maxErr / bins, 1);
    const hist = Array.from({ length: bins }, (_, idx) => ({
      band: `${Math.round(idx * width)}`,
      count: 0,
    }));

    errors.forEach((err) => {
      const idx = Math.min(bins - 1, Math.max(0, Math.floor(err / width)));
      hist[idx].count += 1;
    });

    return hist;
  }, [activeModel]);

  if (isLoading) return <LoadingState label="Loading model comparison..." />;
  if (error) return <ErrorState message={error.message} />;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Model Comparison
        </Typography>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent>
            <Typography variant="subtitle2" color="text.secondary">
              Best Model
            </Typography>
            <Typography variant="h5" color="primary.main" sx={{ mt: 1 }}>
              {data.best_model || "N/A"}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Rows used: {data.rows_used}
            </Typography>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={8}>
        <Card>
          <CardContent>
            <FormControl fullWidth>
              <InputLabel id="model-comparison-select">Model</InputLabel>
              <Select
                labelId="model-comparison-select"
                label="Model"
                value={activeModelName}
                onChange={(event) => setSelectedModel(event.target.value)}
              >
                {modelNames.map((name) => (
                  <MenuItem key={name} value={name}>
                    {name} ({(data?.models || []).find((m) => m.name === name)?.family || "standard"})
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </CardContent>
        </Card>
      </Grid>
      {data?.comparison_summary && (
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Hybrid vs Standard Benchmark
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Best standard: {data.comparison_summary.best_standard_model || "N/A"} (RMSE{" "}
                {data.comparison_summary.best_standard_rmse?.toFixed?.(2) || "-"})
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Best hybrid: {data.comparison_summary.best_hybrid_model || "N/A"} (RMSE{" "}
                {data.comparison_summary.best_hybrid_rmse?.toFixed?.(2) || "-"})
              </Typography>
              {data.comparison_summary.hybrid_rmse_improvement_pct !== null &&
              data.comparison_summary.hybrid_rmse_improvement_pct !== undefined && (
                <Typography
                  variant="body2"
                  color={
                    data.comparison_summary.hybrid_beats_standard
                      ? "success.main"
                      : "text.secondary"
                  }
                  sx={{ mt: 1 }}
                >
                  Hybrid RMSE improvement: {data.comparison_summary.hybrid_rmse_improvement_pct.toFixed(2)}%
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
      )}
      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 300 }}>
            <Typography variant="h6" gutterBottom>
              RMSE vs MAE by Model
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={leaderboard}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rmse" fill="#1f3c88" />
                <Bar dataKey="mae" fill="#0b6e4f" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent sx={{ height: 430 }}>
            <Typography variant="h6" gutterBottom>
              Actual vs Predicted ({activeModelName})
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              R2: {activeModel?.metrics?.r2?.toFixed?.(4) || "-"} | MAE: {activeModel?.metrics?.mae?.toFixed?.(2) || "-"} cycles | RMSE: {activeModel?.metrics?.rmse?.toFixed?.(2) || "-"} cycles
            </Typography>
            <Grid container spacing={1} sx={{ height: "calc(100% - 48px)" }}>
              <Grid item xs={12} sx={{ height: "24%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={parityData.actualHist}>
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
                      domain={[parityData.min, parityData.max]}
                    />
                    <YAxis
                      type="number"
                      dataKey="predicted"
                      name="Predicted RUL"
                      domain={[parityData.min, parityData.max]}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3" }}
                      formatter={(value, name) => [value, name]}
                    />
                    <ReferenceLine
                      segment={[
                        { x: parityData.min, y: parityData.min },
                        { x: parityData.max, y: parityData.max },
                      ]}
                      stroke="#1e293b"
                      strokeWidth={2}
                    />
                    <Scatter data={parityData.points}>
                      {parityData.points.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </Grid>
              <Grid item xs={3} sx={{ height: "76%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={parityData.predictedHist} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" allowDecimals={false} />
                    <YAxis type="category" dataKey="bucket" hide />
                    <Tooltip />
                    <Bar dataKey="count" fill="#5eead4" />
                  </BarChart>
                </ResponsiveContainer>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent sx={{ height: 360 }}>
            <Typography variant="h6" gutterBottom>
              Error Profile ({activeModelName})
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={errorHistogram}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="band" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#fb923c" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 340 }}>
            <Typography variant="h6" gutterBottom>
              Feature Importance (Top 10)
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={importanceSeries} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis type="category" dataKey="feature" width={160} />
                <Tooltip />
                <Bar dataKey="importance" fill="#0077b6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ModelComparison;