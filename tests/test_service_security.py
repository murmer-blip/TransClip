import http.client
import json
import unittest

from tests.service_helpers import serve_test_engine, stop_server


def _request(
    host: str,
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: dict | None = None,
) -> tuple[int, dict[str, str], dict]:
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        send_headers = {"content-type": "application/json"}
        if headers:
            send_headers.update(headers)
        data = json.dumps(body).encode("utf-8") if body is not None else None
        conn.request(method, path, body=data, headers=send_headers)
        response = conn.getresponse()
        raw = response.read().decode("utf-8")
        payload = json.loads(raw) if raw else {}
        return response.status, {k.lower(): v for k, v in response.getheaders()}, payload
    finally:
        conn.close()


class ServiceSecurityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server, self.thread, self.host, self.port = serve_test_engine()
        self.addCleanup(stop_server, self.server, self.thread)

    def test_loopback_request_without_origin_is_allowed(self) -> None:
        status, headers, _ = _request(self.host, self.port, "GET", "/health")
        self.assertEqual(status, 200)
        # No wildcard CORS is ever emitted.
        self.assertNotEqual(headers.get("access-control-allow-origin"), "*")

    def test_foreign_origin_is_rejected(self) -> None:
        status, _headers, payload = _request(
            self.host,
            self.port,
            "POST",
            "/record/start",
            headers={"Origin": "https://evil.example"},
            body={},
        )
        self.assertEqual(status, 403)
        self.assertIn("cross-origin", payload.get("error", ""))

    def test_dns_rebinding_host_is_rejected(self) -> None:
        status, _headers, payload = _request(
            self.host,
            self.port,
            "GET",
            "/health",
            headers={"Host": "attacker.example"},
        )
        self.assertEqual(status, 403)
        self.assertIn("host", payload.get("error", "").lower())

    def test_loopback_origin_is_echoed_not_wildcarded(self) -> None:
        origin = f"http://127.0.0.1:{self.port}"
        status, headers, _ = _request(
            self.host,
            self.port,
            "GET",
            "/health",
            headers={"Origin": origin},
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("access-control-allow-origin"), origin)

    def test_preflight_from_foreign_origin_is_rejected(self) -> None:
        status, _headers, _payload = _request(
            self.host,
            self.port,
            "OPTIONS",
            "/record/start",
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
