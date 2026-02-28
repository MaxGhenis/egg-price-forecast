import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelExplainer } from "../ModelExplainer";

const mockModelInfo = {
  n_obs: 6200,
  n_commodities: 15,
  coefficients: {
    intercept: -0.3,
    seasonal_up_rate: 1.2,
    decline_streak: 0.5,
    recent_momentum: 0.8,
    cpi_expectation: 0.3,
    is_eggs: -0.1,
  },
};

describe("ModelExplainer", () => {
  it("renders the heading", () => {
    render(
      <ModelExplainer
        declineStreak={9}
        seasonalRate={0.3478}
        recentMomentum={-0.05}
        modelInfo={mockModelInfo}
      />
    );
    expect(screen.getByText(/how the model works/i)).toBeInTheDocument();
  });

  it("describes the training data", () => {
    render(
      <ModelExplainer
        declineStreak={9}
        seasonalRate={0.3478}
        recentMomentum={-0.05}
        modelInfo={mockModelInfo}
      />
    );
    expect(screen.getByText(/BLS Average Price/i)).toBeInTheDocument();
    expect(screen.getByText(/15 food commodities/i)).toBeInTheDocument();
  });

  it("explains the model features", () => {
    render(
      <ModelExplainer
        declineStreak={9}
        seasonalRate={0.3478}
        recentMomentum={-0.05}
        modelInfo={mockModelInfo}
      />
    );
    expect(screen.getByText(/1\. Seasonal up rate/i)).toBeInTheDocument();
    expect(screen.getByText(/2\. Decline streak/i)).toBeInTheDocument();
    expect(screen.getByText(/3\. Recent momentum/i)).toBeInTheDocument();
    expect(screen.getByText(/4\. CPI inflation expectations/i)).toBeInTheDocument();
    expect(screen.getByText(/5\. Egg-specific intercept/i)).toBeInTheDocument();
  });

  it("shows the current streak value", () => {
    render(
      <ModelExplainer
        declineStreak={9}
        seasonalRate={0.3478}
        recentMomentum={-0.05}
        modelInfo={mockModelInfo}
      />
    );
    expect(screen.getByText(/= 9 months/)).toBeInTheDocument();
  });

  it("describes the Bayesian approach", () => {
    render(
      <ModelExplainer
        declineStreak={9}
        seasonalRate={0.3478}
        recentMomentum={-0.05}
        modelInfo={mockModelInfo}
      />
    );
    expect(screen.getByText(/Bayesian logistic regression/i)).toBeInTheDocument();
  });
});
