import axios from "axios";

const requestTimeoutMs = Number(process.env.REACT_APP_API_TIMEOUT_MS || 60000);

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || "/",
  timeout: Number.isFinite(requestTimeoutMs) && requestTimeoutMs > 0 ? requestTimeoutMs : 60000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.code === "ECONNABORTED") {
      return Promise.reject(
        new Error("Request timed out. Backend may be waking up on free tier. Please retry in 30-60 seconds.")
      );
    }

    const message =
      error?.response?.data?.message || error.message || "Request failed";
    return Promise.reject(new Error(message));
  }
);

export default apiClient;