import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import DashboardLayout from "./components/DashboardLayout";
import RULPrediction from "./components/RULPrediction";
import DegradationTrends from "./components/DegradationTrends";
import SensitivityAnalysis from "./components/SensitivityAnalysis";
import ModelComparison from "./components/ModelComparison";
import HybridComparison from "./components/HybridComparison";

function App() {
  return (
    <DashboardLayout>
      <Routes>
        <Route path="/rul-prediction" element={<RULPrediction />} />
        <Route path="/degradation-trends" element={<DegradationTrends />} />
        <Route
          path="/sensitivity-analysis"
          element={<SensitivityAnalysis />}
        />
        <Route path="/model-comparison" element={<ModelComparison />} />
        <Route path="/hybrid-comparison" element={<HybridComparison />} />
        <Route path="/" element={<Navigate to="/rul-prediction" replace />} />
      </Routes>
    </DashboardLayout>
  );
}

export default App;