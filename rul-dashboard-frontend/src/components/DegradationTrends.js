import React from "react";
import { useQuery } from "react-query";
import { Card, CardContent, Grid, Typography } from "@mui/material";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getDegradationTrends } from "../api/rulApi";
import { ErrorState, LoadingState } from "./StateViews";

const DegradationTrends = () => {
  const { data, isLoading, error } = useQuery(
    "degradationTrends",
    getDegradationTrends
  );

  if (isLoading) return <LoadingState label="Loading degradation trends..." />;
  if (error) return <ErrorState message={error.message} />;

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Degradation Trends
        </Typography>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent sx={{ height: 420 }}>
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="degradeGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1f3c88" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#1f3c88" stopOpacity={0.1} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="degradation"
                  stroke="#1f3c88"
                  fill="url(#degradeGradient)"
                  strokeWidth={3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default DegradationTrends;