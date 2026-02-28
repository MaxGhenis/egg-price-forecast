import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelExplainer } from "../ModelExplainer";

const mockModelInfo = {
  type: "lightgbm_quantile",
  n_quantile_models: 7,
  n_samples: 1000,
  feature_importances: {
    lag_1: 0.25,
    lag_12: 0.15,
    pct_change_1m: 0.12,
    month_sin: 0.10,
    cpi_expectation: 0.08,
  },
};

const mockBacktest = {
  accuracy: 0.5,
  direction_accuracy: 0.5,
  mape: 0.074,
  coverage_90: 0.808,
  coverage_50: 0.269,
  brier_score: 0.325,
};

describe("ModelExplainer", () => {
  it("renders the heading", () => {
    render(<ModelExplainer modelInfo={mockModelInfo} backtest={mockBacktest} />);
    expect(screen.getByText(/how the model works/i)).toBeInTheDocument();
  });

  it("describes the training data", () => {
    render(<ModelExplainer modelInfo={mockModelInfo} backtest={mockBacktest} />);
    expect(screen.getByText(/BLS Average Price/i)).toBeInTheDocument();
  });

  it("explains the LightGBM approach", () => {
    render(<ModelExplainer modelInfo={mockModelInfo} backtest={mockBacktest} />);
    expect(screen.getByText(/LightGBM multi-quantile regression/i)).toBeInTheDocument();
  });

  it("describes how P(up) is derived", () => {
    render(<ModelExplainer modelInfo={mockModelInfo} backtest={mockBacktest} />);
    expect(screen.getByText(/skew-normal distribution/i)).toBeInTheDocument();
  });

  it("shows backtest metrics", () => {
    render(<ModelExplainer modelInfo={mockModelInfo} backtest={mockBacktest} />);
    expect(screen.getByText(/50%/)).toBeInTheDocument();
    expect(screen.getByText(/7\.4%/)).toBeInTheDocument();
  });
});
