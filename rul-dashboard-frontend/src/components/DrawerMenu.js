import React from "react";
import {
  Box,
  Divider,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
} from "@mui/material";
import TimelineIcon from "@mui/icons-material/Timeline";
import ShowChartIcon from "@mui/icons-material/ShowChart";
import TuneIcon from "@mui/icons-material/Tune";
import InsightsIcon from "@mui/icons-material/Insights";
import BoltIcon from "@mui/icons-material/Bolt";
import { NavLink } from "react-router-dom";

const drawerWidth = 280;

const links = [
  {
    label: "RUL Prediction",
    path: "/rul-prediction",
    icon: <TimelineIcon />,
  },
  {
    label: "Degradation Trends",
    path: "/degradation-trends",
    icon: <ShowChartIcon />,
  },
  {
    label: "Sensitivity Analysis",
    path: "/sensitivity-analysis",
    icon: <TuneIcon />,
  },
  {
    label: "Model Comparison",
    path: "/model-comparison",
    icon: <InsightsIcon />,
  },
  {
    label: "Hybrid Advantage",
    path: "/hybrid-comparison",
    icon: <BoltIcon />,
  },
];

const DrawerMenu = ({ mobileOpen, onClose }) => {
  const drawerContent = (
    <>
      <Toolbar>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          RUL Insight Hub
        </Typography>
      </Toolbar>
      <Divider />
      <List sx={{ p: 1 }}>
        {links.map((item) => (
          <ListItemButton
            key={item.path}
            component={NavLink}
            to={item.path}
            onClick={onClose}
            sx={{
              borderRadius: 2,
              mb: 0.5,
              "&.active": {
                bgcolor: "primary.main",
                color: "white",
                "& .MuiListItemIcon-root": { color: "white" },
              },
            }}
          >
            <ListItemIcon>{item.icon}</ListItemIcon>
            <ListItemText primary={item.label} />
          </ListItemButton>
        ))}
      </List>
    </>
  );

  return (
    <Box component="nav" sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}>
      <Drawer
        variant="temporary"
        open={mobileOpen}
        onClose={onClose}
        ModalProps={{ keepMounted: true }}
        sx={{
          display: { xs: "block", md: "none" },
          "& .MuiDrawer-paper": {
            boxSizing: "border-box",
            width: drawerWidth,
          },
        }}
      >
        {drawerContent}
      </Drawer>
      <Drawer
        variant="permanent"
        sx={{
          display: { xs: "none", md: "block" },
          "& .MuiDrawer-paper": {
            boxSizing: "border-box",
            width: drawerWidth,
          },
        }}
        open
      >
        {drawerContent}
      </Drawer>
    </Box>
  );
};

export { drawerWidth };
export default DrawerMenu;