import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "light",
    primary: {
      main: "#0b6e4f",
    },
    secondary: {
      main: "#1f3c88",
    },
    background: {
      default: "#f4f7fb",
      paper: "#ffffff",
    },
  },
  shape: {
    borderRadius: 12,
  },
  typography: {
    fontFamily: "'Trebuchet MS', 'Segoe UI', sans-serif",
    h4: {
      fontWeight: 700,
      letterSpacing: 0.4,
    },
  },
});

export default theme;