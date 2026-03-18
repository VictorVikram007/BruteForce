import React from "react";
import { Alert, Box, CircularProgress, Typography } from "@mui/material";

export const LoadingState = ({ label = "Loading data..." }) => (
  <Box
    sx={{
      minHeight: 220,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      gap: 2,
    }}
  >
    <CircularProgress size={28} />
    <Typography>{label}</Typography>
  </Box>
);

export const ErrorState = ({ message }) => (
  <Alert severity="error" sx={{ mt: 2 }}>
    {message || "Unable to load data right now."}
  </Alert>
);