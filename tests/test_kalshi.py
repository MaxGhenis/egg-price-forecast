"""Tests for egg_price_forecast.kalshi."""

import base64
import json
import os
import tempfile

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization


@pytest.fixture
def rsa_key_pair():
    """Generate a test RSA key pair and save to a temp file."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".pem", delete=False
    ) as f:
        f.write(pem)
        key_path = f.name

    yield {"private_key": private_key, "key_path": key_path}

    os.unlink(key_path)


@pytest.fixture
def client(rsa_key_pair):
    """Create a KalshiClient with a test RSA key."""
    from egg_price_forecast.kalshi import KalshiClient

    return KalshiClient(
        api_key_id="test-api-key-id",
        private_key_path=rsa_key_pair["key_path"],
    )


class TestKalshiClientInit:
    """Tests for KalshiClient initialization."""

    def test_init_loads_key(self, rsa_key_pair):
        """Client initializes and loads the RSA private key."""
        from egg_price_forecast.kalshi import KalshiClient

        client = KalshiClient(
            api_key_id="test-key",
            private_key_path=rsa_key_pair["key_path"],
        )
        assert client.api_key_id == "test-key"
        assert client._private_key is not None

    def test_init_invalid_key_path_raises(self):
        """Client raises error with invalid key path."""
        from egg_price_forecast.kalshi import KalshiClient

        with pytest.raises(Exception):
            KalshiClient(
                api_key_id="test-key",
                private_key_path="/nonexistent/path/key.pem",
            )

    def test_base_url(self, client):
        """Client has the correct base URL."""
        assert (
            client.BASE_URL
            == "https://api.elections.kalshi.com/trade-api/v2"
        )


class TestSignRequest:
    """Tests for _sign_request method."""

    def test_produces_valid_base64(self, client):
        """Signature is a valid base64-encoded string."""
        signature = client._sign_request("GET", "/trade-api/v2/markets/TEST", 1700000000)
        # Should not raise
        decoded = base64.b64decode(signature)
        assert len(decoded) > 0

    def test_different_inputs_different_signatures(self, client):
        """Different inputs produce different signatures."""
        sig1 = client._sign_request("GET", "/trade-api/v2/markets/A", 1700000000)
        sig2 = client._sign_request("GET", "/trade-api/v2/markets/B", 1700000000)
        assert sig1 != sig2

    def test_signature_can_be_verified(self, client, rsa_key_pair):
        """The signature can be verified with the corresponding public key."""
        timestamp = 1700000000
        method = "GET"
        path = "/trade-api/v2/markets/TEST"
        signature = client._sign_request(method, path, timestamp)

        decoded_sig = base64.b64decode(signature)
        public_key = rsa_key_pair["private_key"].public_key()

        message = f"{timestamp}{method}{path}".encode()
        # Should not raise if signature is valid
        public_key.verify(
            decoded_sig,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )


class TestHeaders:
    """Tests for _headers method."""

    def test_headers_contain_required_keys(self, client):
        """Headers include all required Kalshi auth headers."""
        headers = client._headers("GET", "/trade-api/v2/markets/TEST")
        assert "KALSHI-ACCESS-KEY" in headers
        assert "KALSHI-ACCESS-SIGNATURE" in headers
        assert "KALSHI-ACCESS-TIMESTAMP" in headers

    def test_headers_key_matches_api_key_id(self, client):
        """KALSHI-ACCESS-KEY matches the api_key_id."""
        headers = client._headers("GET", "/trade-api/v2/markets/TEST")
        assert headers["KALSHI-ACCESS-KEY"] == "test-api-key-id"

    def test_headers_timestamp_is_string(self, client):
        """Timestamp in headers is a string representation of an integer."""
        headers = client._headers("GET", "/trade-api/v2/markets/TEST")
        ts = headers["KALSHI-ACCESS-TIMESTAMP"]
        assert isinstance(ts, str)
        int(ts)  # Should not raise


class TestGetMarket:
    """Tests for get_market method."""

    def test_get_market_calls_correct_url(self, client, mocker):
        """get_market sends GET request to correct endpoint."""
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "market": {
                "ticker": "KXEGGS-FEB26-0",
                "yes_bid": 31,
                "no_bid": 65,
                "status": "active",
            }
        }
        mock_response.raise_for_status = mocker.Mock()
        mock_get = mocker.patch("requests.get", return_value=mock_response)

        result = client.get_market("KXEGGS-FEB26-0")

        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "KXEGGS-FEB26-0" in call_url
        assert result["market"]["ticker"] == "KXEGGS-FEB26-0"

    def test_get_market_returns_dict(self, client, mocker):
        """get_market returns parsed JSON dict."""
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "market": {"ticker": "TEST", "yes_bid": 50}
        }
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch("requests.get", return_value=mock_response)

        result = client.get_market("TEST")
        assert isinstance(result, dict)
        assert "market" in result


class TestComputeEdge:
    """Tests for compute_edge method."""

    def test_positive_edge_buy_yes(self, client):
        """Positive edge recommends BUY YES."""
        market = {"market": {"yes_bid": 30}}  # 0.30
        result = client.compute_edge(0.60, market)

        assert result["edge"] == pytest.approx(0.30)
        assert result["kelly_fraction"] > 0
        assert result["ev_per_dollar"] > 0
        assert result["recommendation"] == "BUY YES"

    def test_negative_edge_buy_no(self, client):
        """Negative edge recommends BUY NO."""
        market = {"market": {"yes_bid": 70}}  # 0.70
        result = client.compute_edge(0.30, market)

        assert result["edge"] == pytest.approx(-0.40)
        assert result["kelly_fraction"] == 0
        assert result["ev_per_dollar"] < 0
        assert result["recommendation"] == "BUY NO"

    def test_zero_edge(self, client):
        """Zero edge: no recommendation."""
        market = {"market": {"yes_bid": 50}}  # 0.50
        result = client.compute_edge(0.50, market)

        assert result["edge"] == pytest.approx(0.0)
        assert result["kelly_fraction"] == 0
        assert result["recommendation"] == "BUY NO"

    def test_edge_calculation_values(self, client):
        """Test specific numeric values for edge calculation."""
        market = {"market": {"yes_bid": 25}}  # yes_price = 0.25
        result = client.compute_edge(0.50, market)

        # edge = 0.50 - 0.25 = 0.25
        assert result["edge"] == pytest.approx(0.25)
        # kelly = 0.25 / (1 - 0.25) = 0.25 / 0.75 = 0.3333
        assert result["kelly_fraction"] == pytest.approx(1 / 3, rel=1e-3)
        # ev_per_dollar = 0.25 * (1/0.25 - 1) = 0.25 * 3 = 0.75
        assert result["ev_per_dollar"] == pytest.approx(0.75)

    def test_edge_with_very_low_yes_price(self, client):
        """Edge calculation with very low yes price."""
        market = {"market": {"yes_bid": 5}}  # yes_price = 0.05
        result = client.compute_edge(0.50, market)

        assert result["edge"] == pytest.approx(0.45)
        # kelly = 0.45 / 0.95 ≈ 0.4737
        assert result["kelly_fraction"] == pytest.approx(0.45 / 0.95, rel=1e-3)
        assert result["recommendation"] == "BUY YES"

    def test_edge_returns_correct_structure(self, client):
        """compute_edge returns dict with all required keys."""
        market = {"market": {"yes_bid": 40}}
        result = client.compute_edge(0.50, market)

        assert "edge" in result
        assert "kelly_fraction" in result
        assert "ev_per_dollar" in result
        assert "recommendation" in result
