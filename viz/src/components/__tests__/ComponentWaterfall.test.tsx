import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComponentWaterfall } from "../ComponentWaterfall";

const mockComponents = {
  lag_1: 0.25,
  lag_12: 0.15,
  pct_change_1m: 0.12,
  cpi_expectation: 0.08,
  cross_commodity_momentum: -0.05,
  is_eggs: -0.02,
};

describe("ComponentWaterfall", () => {
  it("renders all component labels", () => {
    render(<ComponentWaterfall components={mockComponents} total={0.65} />);
    expect(screen.getByText(/last month price/i)).toBeInTheDocument();
    expect(screen.getByText(/same month last year/i)).toBeInTheDocument();
    expect(screen.getByText(/1-month momentum/i)).toBeInTheDocument();
    expect(screen.getByText(/cpi expectations/i)).toBeInTheDocument();
    expect(screen.getByText(/cross-commodity/i)).toBeInTheDocument();
    expect(screen.getAllByText(/egg-specific/i).length).toBeGreaterThan(0);
  });

  it("renders the P(up) total row", () => {
    render(<ComponentWaterfall components={mockComponents} total={0.65} />);
    expect(screen.getByText("P(up)")).toBeInTheDocument();
    expect(screen.getByText("65.0%")).toBeInTheDocument();
  });

  it("shows component values with sign", () => {
    render(<ComponentWaterfall components={mockComponents} total={0.65} />);
    expect(screen.getByText("+0.25")).toBeInTheDocument();
    expect(screen.getByText("-0.05")).toBeInTheDocument();
  });

  it("marks near-zero components as inactive", () => {
    const withZero = { ...mockComponents, is_eggs: 0.0 };
    const { container } = render(
      <ComponentWaterfall components={withZero} total={0.65} />
    );
    expect(container.querySelector("[data-inactive]")).toBeTruthy();
  });
});
