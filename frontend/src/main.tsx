import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router";

import { App } from "./App";
import { EchoRoute } from "./routes/EchoRoute";
import { HealthRoute } from "./routes/HealthRoute";
import "./styles.css";

const queryClient = new QueryClient();

const container = document.getElementById("root");
if (container === null) {
  throw new Error("index.html is missing its #root element.");
}

createRoot(container).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<App />}>
            <Route index element={<HealthRoute />} />
            <Route path="echo" element={<EchoRoute />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
