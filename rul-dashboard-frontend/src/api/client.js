import axios from "axios";

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_BASE_URL || "/",
  timeout: 10000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error?.response?.data?.message || error.message || "Request failed";
    return Promise.reject(new Error(message));
  }
);

export default apiClient;