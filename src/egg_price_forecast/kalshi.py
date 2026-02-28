"""Kalshi API client with RSA-PSS authentication."""

from __future__ import annotations

import base64
import time

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class KalshiClient:
    """Client for the Kalshi trading API with RSA-PSS authentication."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key_id: str, private_key_path: str):
        """Initialize with API key ID and path to RSA private key PEM file.

        Args:
            api_key_id: Kalshi API key identifier.
            private_key_path: Path to PEM-encoded RSA private key file.

        Raises:
            FileNotFoundError: If key file does not exist.
            ValueError: If key file is not a valid RSA private key.
        """
        self.api_key_id = api_key_id
        with open(private_key_path, "rb") as f:
            self._private_key = serialization.load_pem_private_key(
                f.read(), password=None
            )

    def _sign_request(self, method: str, path: str, timestamp: int) -> str:
        """Sign a request using RSA-PSS with SHA256.

        Signs the string "{timestamp}{method}{path}".

        Args:
            method: HTTP method (e.g., "GET", "POST").
            path: Request path (e.g., "/trade-api/v2/markets/TICKER").
            timestamp: Unix timestamp as integer.

        Returns:
            Base64-encoded signature string.
        """
        message = f"{timestamp}{method}{path}".encode()
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def _headers(self, method: str, path: str) -> dict:
        """Build authentication headers for a Kalshi API request.

        Args:
            method: HTTP method.
            path: Request path.

        Returns:
            Dict with KALSHI-ACCESS-KEY, KALSHI-ACCESS-SIGNATURE,
            and KALSHI-ACCESS-TIMESTAMP headers.
        """
        timestamp = int(time.time())
        signature = self._sign_request(method, path, timestamp)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
        }

    def get_market(self, ticker: str) -> dict:
        """Fetch market data for a given ticker.

        Args:
            ticker: Market ticker (e.g., "KXEGGS-FEB26-0").

        Returns:
            Parsed JSON response dict containing market data.
        """
        path = f"/trade-api/v2/markets/{ticker}"
        url = f"{self.BASE_URL.rstrip('/trade-api/v2')}{path}"
        # Reconstruct full URL properly
        url = f"https://api.elections.kalshi.com{path}"
        headers = self._headers("GET", path)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    def compute_edge(self, estimated_prob: float, market: dict) -> dict:
        """Compute trading edge given model probability and market data.

        Args:
            estimated_prob: Model's estimated probability (0 to 1).
            market: Market data dict from get_market (contains
                    market['market']['yes_bid'] in cents).

        Returns:
            Dict with 'edge', 'kelly_fraction', 'ev_per_dollar',
            and 'recommendation'.
        """
        yes_price = market["market"]["yes_bid"] / 100.0
        edge = estimated_prob - yes_price

        if edge > 0:
            kelly_fraction = edge / (1 - yes_price)
            ev_per_dollar = edge * (1.0 / yes_price - 1)
            recommendation = "BUY YES"
        else:
            kelly_fraction = 0.0
            ev_per_dollar = edge * (1.0 / yes_price - 1) if yes_price > 0 else 0.0
            recommendation = "BUY NO"

        return {
            "edge": edge,
            "kelly_fraction": kelly_fraction,
            "ev_per_dollar": ev_per_dollar,
            "recommendation": recommendation,
        }
