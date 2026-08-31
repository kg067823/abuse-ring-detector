"""Small API client for the AbuseRing Command Center."""
from __future__ import annotations

import os
from typing import Any

import requests


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class CommandCenterClient:
    def __init__(self, base_url: str | None = None, token: str | None = None, timeout: float = 10.0):
        self.base_url = (base_url or os.getenv("ABUSERING_API_URL", "http://localhost:8000")).rstrip("/")
        self.token = token or os.getenv("ABUSERING_ADMIN_TOKEN", "")
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = requests.request(method, self.base_url + path, headers=headers, timeout=self.timeout, **kwargs)
        except requests.RequestException as exc:
            raise ApiError(f"API unavailable: {exc}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(str(detail) or f"HTTP {response.status_code}", response.status_code)
        try:
            return response.json()
        except ValueError as exc:
            raise ApiError("API returned invalid JSON", response.status_code) from exc

    def health(self): return self._request("GET", "/health")
    def readiness(self): return self._request("GET", "/readiness")
    def liveness(self): return self._request("GET", "/liveness")
    def metrics(self):
        try:
            response = __import__("requests").get(self.base_url + "/metrics", timeout=self.timeout)
            if response.status_code >= 400:
                raise ApiError(f"HTTP {response.status_code}", response.status_code)
            return response.text
        except __import__("requests").RequestException as exc:
            raise ApiError(f"API unavailable: {exc}") from exc
    def predict(self, payload: dict[str, Any]): return self._request("POST", "/v1/predict", json=payload, headers={"X-Correlation-ID": payload.get("order_id", "")})
    def alerts(self, min_risk: float | None = None):
        return self._request("GET", "/v1/alerts", params={"min_risk": min_risk} if min_risk is not None else {})
    def cases(self, status: str | None = None, severity: str | None = None, min_risk: float | None = None):
        params = {k: v for k, v in {"status_filter": status, "severity": severity, "min_risk": min_risk}.items() if v is not None}
        return self._request("GET", "/v1/cases", params=params)
    def case(self, case_id: str): return self._request("GET", f"/v1/cases/{case_id}")
    def graph(self, case_id: str): return self._request("GET", f"/v1/cases/{case_id}/graph")
    def timeline(self, case_id: str): return self._request("GET", f"/v1/cases/{case_id}/timeline")
    def evidence(self, case_id: str): return self._request("GET", f"/v1/cases/{case_id}/evidence")
    def update_status(self, case_id: str, status: str, actor: str, reason: str):
        return self._request("PATCH", f"/v1/cases/{case_id}/status", json={"status": status, "actor": actor, "reason": reason})
    def add_note(self, case_id: str, note: str, actor: str):
        return self._request("POST", f"/v1/cases/{case_id}/notes", json={"note": note, "actor": actor})
