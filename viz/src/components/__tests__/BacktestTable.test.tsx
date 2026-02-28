import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BacktestTable } from "../BacktestTable";

const mockBacktest = {
  accuracy: 0.5385,
  brier_score: 0.2777,
  predictions: [
    { year: 2020, predicted_prob: 0.35, actual_up: false, correct: true },
    { year: 2021, predicted_prob: 0.42, actual_up: true, correct: false },
    { year: 2022, predicted_prob: 0.55, actual_up: true, correct: true },
    { year: 2023, predicted_prob: 0.38, actual_up: false, correct: true },
    { year: 2024, predicted_prob: 0.45, actual_up: false, correct: true },
    { year: 2025, predicted_prob: 0.6, actual_up: true, correct: true },
  ],
};

describe("BacktestTable", () => {
  it("renders accuracy metric", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText(/53\.8%/)).toBeInTheDocument();
  });

  it("renders brier score", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    expect(screen.getByText(/0\.278/)).toBeInTheDocument();
  });

  it("renders prediction rows", () => {
    render(<BacktestTable backtest={mockBacktest} />);
    // Dot matrix shows 2-digit years
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
});
