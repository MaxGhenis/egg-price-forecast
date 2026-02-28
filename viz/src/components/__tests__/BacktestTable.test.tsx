import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BacktestTable } from "../BacktestTable";

const mockBacktest = {
  accuracy: 0.5385,
  direction_accuracy: 0.5385,
  mape: 0.074,
  coverage_90: 0.808,
  coverage_50: 0.269,
  brier_score: 0.2777,
  predictions: [
    { year: 2020, month: 1, date: "2020-01-01", predicted_prob: 0.35, predicted_price: 1.50, actual_price: 1.45, current_price: 1.48, actual_up: false, correct: true, error: 0.05, in_90_interval: true, in_50_interval: true },
    { year: 2021, month: 1, date: "2021-01-01", predicted_prob: 0.42, predicted_price: 1.60, actual_price: 1.70, current_price: 1.55, actual_up: true, correct: false, error: 0.10, in_90_interval: true, in_50_interval: false },
    { year: 2022, month: 1, date: "2022-01-01", predicted_prob: 0.55, predicted_price: 1.80, actual_price: 1.85, current_price: 1.75, actual_up: true, correct: true, error: 0.05, in_90_interval: true, in_50_interval: true },
    { year: 2023, month: 1, date: "2023-01-01", predicted_prob: 0.38, predicted_price: 2.00, actual_price: 1.95, current_price: 2.05, actual_up: false, correct: true, error: 0.05, in_90_interval: true, in_50_interval: true },
    { year: 2024, month: 1, date: "2024-01-01", predicted_prob: 0.45, predicted_price: 2.10, actual_price: 2.05, current_price: 2.15, actual_up: false, correct: true, error: 0.05, in_90_interval: true, in_50_interval: true },
    { year: 2025, month: 1, date: "2025-01-01", predicted_prob: 0.6, predicted_price: 2.50, actual_price: 2.55, current_price: 2.45, actual_up: true, correct: true, error: 0.05, in_90_interval: true, in_50_interval: true },
  ],
};

describe("BacktestTable", () => {
  it("renders direction accuracy metric", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText(/53\.8%/)).toBeInTheDocument();
  });

  it("renders brier score", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText(/0\.278/)).toBeInTheDocument();
  });

  it("renders prediction rows", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText("20")).toBeInTheDocument();
    expect(screen.getByText("25")).toBeInTheDocument();
  });

  it("marks correct predictions", () => {
    const { container } = render(<BacktestTable backtest={mockBacktest} />);
    const correctCells = container.querySelectorAll("[data-correct='true']");
    expect(correctCells.length).toBe(5);
  });

  it("marks incorrect predictions", () => {
    const { container } = render(<BacktestTable backtest={mockBacktest} />);
    const wrongCells = container.querySelectorAll("[data-correct='false']");
    expect(wrongCells.length).toBe(1);
  });

  it("renders MAPE metric", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText(/7\.4%/)).toBeInTheDocument();
  });

  it("renders 90% coverage metric", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText(/81%/)).toBeInTheDocument();
  });
});
