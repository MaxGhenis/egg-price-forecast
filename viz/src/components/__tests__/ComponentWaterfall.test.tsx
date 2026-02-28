import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComponentWaterfall } from "../ComponentWaterfall";

const mockComponents = {
  intercept: -0.3,
  seasonal: 0.52,
  streak: 0.15,
  momentum: -0.08,
  cpi_signal: 0.03,
  egg_specific: -0.05,
};

describe("ComponentWaterfall", () => {
  it("renders all component labels", () => {
    render(<ComponentWaterfall components={mockComponents} total={0.553} />);
    expect(screen.getByText(/baseline/i)).toBeInTheDocument();
    expect(screen.getByText(/seasonal pattern/i)).toBeInTheDocument();
    expect(screen.getByText(/decline streak/i)).toBeInTheDocument();
    expect(screen.getByText(/recent momentum/i)).toBeInTheDocument();
    expect(screen.getByText(/cpi expectations/i)).toBeInTheDocument();
    expect(screen.getByText(/egg-specific/i)).toBeInTheDocument();
  });

  it("renders the total row", () => {
    render(<ComponentWaterfall components={mockComponents} total={0.553} />);
    expect(screen.getByText(/total/i)).toBeInTheDocument();
    expect(screen.getByText("55.3%")).toBeInTheDocument();
  });

  it("shows component values with sign", () => {
    render(<ComponentWaterfall components={mockComponents} total={0.553} />);
    expect(screen.getByText("+0.52")).toBeInTheDocument();
    expect(screen.getByText("-0.08")).toBeInTheDocument();
  });

  it("marks near-zero components as inactive", () => {
    const withZero = { ...mockComponents, cpi_signal: 0.0 };
    const { container } = render(
      <ComponentWaterfall components={withZero} total={0.553} />
    );
    expect(container.querySelector("[data-inactive]")).toBeTruthy();
  });
});
