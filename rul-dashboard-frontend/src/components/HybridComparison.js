import React, { useMemo } from "react";
import { useQuery } from "react-query";
import {
  Alert,
  Card,
  CardContent,
  Grid,
  Stack,
  Typography,
} from "@mui/material";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getModelComparison } from "../api/rulApi";
import { ErrorState, LoadingState } from "./StateViews";

const HybridComparison = () => {
  const { data, isLoading, error } = useQuery("hybridComparison", getModelComparison);

  const parsed = useMemo(() => {
    const all = data?.models || [];
    const standards = all.filter((model) => (model.family || "standard") === "standard");
    const hybrids = all.filter((model) => model.family === "hybrid");

    if (!standards.length || !hybrids.length) {
      return {
        standards,
        hybrids,
        baseline: null,
        bars: [],
        bestHybrid: null,
        headToHead: [],
      };
    }

    const baseline = [...standards].sort((a, b) => a.metrics.rmse - b.metrics.rmse)[0];
    const bars = hybrids.map((model) => {
      const rmseGainPct =
        ((baseline.metrics.rmse - model.metrics.rmse) / Math.max(baseline.metrics.rmse, 1e-9)) *
        100;
      const maeGainPct =
        ((baseline.metrics.mae - model.metrics.mae) / Math.max(baseline.metrics.mae, 1e-9)) *
        100;
      const r2Delta = model.metrics.r2 - baseline.metrics.r2;

      return {
        name: model.name,
        rmseGainPct: Number(rmseGainPct.toFixed(2)),
        maeGainPct: Number(maeGainPct.toFixed(2)),
        r2Delta: Number(r2Delta.toFixed(4)),
        rmse: Number(model.metrics.rmse.toFixed(3)),
        mae: Number(model.metrics.mae.toFixed(3)),
        r2: Number(model.metrics.r2.toFixed(4)),
      };
    });

    const bestHybrid = [...bars].sort((a, b) => b.rmseGainPct - a.rmseGainPct)[0];

    const headToHead = [
      {
        group: "Best Standard",
        name: baseline.name,
        rmse: Number(baseline.metrics.rmse.toFixed(3)),
        mae: Number(baseline.metrics.mae.toFixed(3)),
        r2: Number(baseline.metrics.r2.toFixed(4)),
      },
      {
        group: "Best Hybrid",
        name: bestHybrid.name,
        rmse: Number(bestHybrid.rmse.toFixed(3)),
        mae: Number(bestHybrid.mae.toFixed(3)),
        r2: Number(bestHybrid.r2.toFixed(4)),
      },
    ];

    return {
      standards,
      hybrids,
      baseline,
      bars,
      bestHybrid,
      headToHead,
    };
  }, [data]);

  if (isLoading) return <LoadingState label="Loading hybrid comparison..." />;
  if (error) return <ErrorState message={error.message} />;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Hybrid Model Advantage
        </Typography>
      </Grid>

      {!parsed.baseline && (
        <Grid item xs={12}>
          <Alert severity="warning">
            Hybrid or standard model groups are missing from the current results.
          </Alert>
        </Grid>
      )}

      {parsed.baseline && parsed.bestHybrid && (
        <>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Baseline (Best Standard)
                </Typography>
                <Typography variant="h6" sx={{ mt: 1 }}>
                  {parsed.baseline.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  RMSE {parsed.baseline.metrics.rmse.toFixed(3)} | MAE {parsed.baseline.metrics.mae.toFixed(3)} | R2{" "}
                  {parsed.baseline.metrics.r2.toFixed(4)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Best Hybrid
                </Typography>
                <Typography variant="h6" sx={{ mt: 1 }}>
                  {parsed.bestHybrid.name}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  RMSE {parsed.bestHybrid.rmse} | MAE {parsed.bestHybrid.mae} | R2 {parsed.bestHybrid.r2}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary">
                  Gain vs Best Standard
                </Typography>
                <Stack spacing={0.5} sx={{ mt: 1 }}>
                  <Typography variant="h6" color="success.main">
                    RMSE gain: {parsed.bestHybrid.rmseGainPct}%
                  </Typography>
                  <Typography variant="body2" color="success.main">
                    MAE gain: {parsed.bestHybrid.maeGainPct}%
                  </Typography>
                  <Typography variant="body2" color={parsed.bestHybrid.r2Delta >= 0 ? "success.main" : "error.main"}>
                    R2 delta: {parsed.bestHybrid.r2Delta >= 0 ? "+" : ""}
                    {parsed.bestHybrid.r2Delta}
                  </Typography>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        </>
      )}

      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 340 }}>
            <Typography variant="h6" gutterBottom>
              Hybrid vs Standard (Head-to-Head)
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={parsed.headToHead}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="group" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="rmse" name="RMSE" fill="#1f3c88" />
                <Bar yAxisId="left" dataKey="mae" name="MAE" fill="#2a9d8f" />
                <Line yAxisId="right" type="monotone" dataKey="r2" name="R2" stroke="#e76f51" strokeWidth={3} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 360 }}>
            <Typography variant="h6" gutterBottom>
              RMSE and MAE Gain (%) by Hybrid Model
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={parsed.bars}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rmseGainPct" name="RMSE Gain %" fill="#2a9d8f" />
                <Bar dataKey="maeGainPct" name="MAE Gain %" fill="#1f3c88" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 320 }}>
            <Typography variant="h6" gutterBottom>
              R2 Delta vs Best Standard
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={parsed.bars}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="r2Delta" name="R2 Delta" fill="#e76f51" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default HybridComparison;
