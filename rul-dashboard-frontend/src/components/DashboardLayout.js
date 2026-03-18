import React, { useState } from "react";
import {
  AppBar,
  Box,
  IconButton,
  Toolbar,
  Typography,
  useMediaQuery,
  useTheme,
} from "@mui/material";
import MenuIcon from "@mui/icons-material/Menu";
import DrawerMenu, { drawerWidth } from "./DrawerMenu";

const DashboardLayout = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const theme = useTheme();
  const isDesktop = useMediaQuery(theme.breakpoints.up("md"));

  const handleDrawerToggle = () => {
    setMobileOpen((prev) => !prev);
  };

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <AppBar
        position="fixed"
        color="inherit"
        elevation={0}
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          borderBottom: "1px solid",
          borderColor: "divider",
          backdropFilter: "blur(8px)",
          bgcolor: "rgba(255,255,255,0.88)",
        }}
      >
        <Toolbar>
          {!isDesktop && (
            <IconButton
              color="inherit"
              edge="start"
              onClick={handleDrawerToggle}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          <Typography variant="h6" sx={{ fontWeight: 700 }}>
            Remaining Useful Life Dashboard
          </Typography>
        </Toolbar>
      </AppBar>

      <DrawerMenu mobileOpen={mobileOpen} onClose={handleDrawerToggle} />

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, sm: 3 },
          ml: { md: `${drawerWidth}px` },
          mt: "64px",
          width: { md: `calc(100% - ${drawerWidth}px)` },
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default DashboardLayout;