import React, { useMemo } from "react";
import { useQuery } from "react-query";
import {
  Card,
  CardContent,
  Grid,
  Typography,
} from "@mui/material";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getSensitivityAnalysis } from "../api/rulApi";
import { ErrorState, LoadingState } from "./StateViews";

const SensitivityAnalysis = () => {
  const { data, isLoading, error } = useQuery(
    "sensitivityAnalysis",
    getSensitivityAnalysis
  );

  const parameters = useMemo(() => data?.parameters || [], [data]);

  const byName = useMemo(
    () =>
      Object.fromEntries(
        parameters.map((param) => [param.name, param.series || []])
      ),
    [parameters]
  );
  const emptySeries = useMemo(() => [], []);

  const temperatureSeries = byName.temperature || emptySeries;
  const dodSeries = byName.depth_of_discharge || emptySeries;
  const chargingSeries = byName.charging_rate || emptySeries;

  const parameterSummary = useMemo(() => {
    const summarize = (name, series) => {
      const impacts = series.map((point) => point.impact);
      const avgImpact =
        impacts.length > 0
          ? impacts.reduce((acc, item) => acc + item, 0) / impacts.length
          : 0;
      const peakImpact = impacts.length > 0 ? Math.max(...impacts) : 0;
      return {
        name,
        avgImpact: Number(avgImpact.toFixed(3)),
        peakImpact: Number(peakImpact.toFixed(3)),
      };
    };

    return [
      summarize("Temperature", temperatureSeries),
      summarize("DoD", dodSeries),
      summarize("Charging Rate", chargingSeries),
    ];
  }, [chargingSeries, dodSeries, temperatureSeries]);

  const normalizedLines = useMemo(() => {
    const build = (series, key) => {
      const maxVal = Math.max(...series.map((point) => point.impact), 1e-6);
      return series.map((point) => ({
        input: point.input,
        [key]: Number((100 * (point.impact / maxVal)).toFixed(1)),
      }));
    };

    const temp = build(temperatureSeries, "temperature");
    const dod = build(dodSeries, "dod");
    const charge = build(chargingSeries, "charge");

    const len = Math.max(temp.length, dod.length, charge.length);
    const rows = [];
    for (let i = 0; i < len; i += 1) {
      rows.push({
        step: i + 1,
        temperature: temp[i]?.temperature ?? null,
        depthOfDischarge: dod[i]?.dod ?? null,
        chargingRate: charge[i]?.charge ?? null,
      });
    }
    return rows;
  }, [chargingSeries, dodSeries, temperatureSeries]);

  if (isLoading) return <LoadingState label="Loading sensitivity profile..." />;
  if (error) return <ErrorState message={error.message} />;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Sensitivity Analysis
        </Typography>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent sx={{ height: 300 }}>
            <Typography variant="h6" gutterBottom>
              Temperature Impact
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={temperatureSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="input" unit="C" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="impact" stroke="#d62828" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent sx={{ height: 300 }}>
            <Typography variant="h6" gutterBottom>
              DoD Impact
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dodSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="input" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="impact" stroke="#1d3557" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={4}>
        <Card>
          <CardContent sx={{ height: 300 }}>
            <Typography variant="h6" gutterBottom>
              Charging Rate Impact
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chargingSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="input" />
                <YAxis />
                <Tooltip />
                <Line type="monotone" dataKey="impact" stroke="#2a9d8f" strokeWidth={3} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent sx={{ height: 320 }}>
            <Typography variant="h6" gutterBottom>
              Peak vs Average Impact by Parameter
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={parameterSummary}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="peakImpact" fill="#0b6e4f" />
                <Bar dataKey="avgImpact" fill="#0077b6" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent sx={{ height: 360 }}>
            <Typography variant="h6" gutterBottom>
              Relative Stress Shape (Normalized)
            </Typography>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={normalizedLines}
                margin={{ top: 12, right: 18, left: 0, bottom: 42 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" />
                <YAxis unit="%" tickFormatter={(value) => `${value}%`} />
                <Tooltip />
                <Legend verticalAlign="bottom" height={36} />
                <Line
                  type="monotone"
                  dataKey="temperature"
                  name="temperature"
                  stroke="#d62828"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="depthOfDischarge"
                  name="depth of discharge"
                  stroke="#1d3557"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="chargingRate"
                  name="charging rate"
                  stroke="#2a9d8f"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Interpretation
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Temperature rises above 40C accelerate degradation sharply, deep DoD cycles above 80% increase wear, and charging above ~1.5C causes the steepest late-stage impact. Use this view to tune thermal control, charging policy, and depth-of-discharge limits.
            </Typography>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default SensitivityAnalysis;