import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProbabilityGauge } from "../ProbabilityGauge";

describe("ProbabilityGauge", () => {
  it("renders the probability percentage", () => {
    render(<ProbabilityGauge probability={0.553} />);
    expect(screen.getByText("55.3%")).toBeInTheDocument();
  });

  it("renders the title", () => {
    render(<ProbabilityGauge probability={0.553} />);
    expect(
      screen.getByText(/probability.*increase/i)
    ).toBeInTheDocument();
  });

  it("applies bullish styling when probability > 0.5", () => {
    const { container } = render(
      <ProbabilityGauge probability={0.65} />
    );
    expect(container.querySelector("[data-signal='bullish']")).toBeTruthy();
  });

  it("applies bearish styling when probability < 0.5", () => {
    const { container } = render(
      <ProbabilityGauge probability={0.35} />
    );
    expect(container.querySelector("[data-signal='bearish']")).toBeTruthy();
  });
});
