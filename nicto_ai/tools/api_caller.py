"""
NICTO AI - API Caller Tool
Make HTTP requests to REST APIs.
"""

import json
import logging
from typing import Dict, Optional
from .base import Tool, ToolResult, ToolParameter, ToolPermission

logger = logging.getLogger(__name__)


class ApiCallerTool(Tool):
    """
    HTTP API client.

    Features:
    - GET, POST, PUT, DELETE, PATCH requests
    - Custom headers
    - JSON body support
    - Query parameters
    - Response parsing
    """

    name = "api_caller"
    description = "Make HTTP API requests (GET, POST, PUT, DELETE). Send headers, body, and query parameters."
    parameters = [
        ToolParameter(name="url", type="string", description="API endpoint URL", required=True),
        ToolParameter(name="method", type="string", description="HTTP method", required=False, default="GET", enum=[
            "GET", "POST", "PUT", "DELETE", "PATCH",
        ]),
        ToolParameter(name="headers", type="string", description="Request headers (JSON string)", required=False),
        ToolParameter(name="body", type="string", description="Request body (JSON string)", required=False),
        ToolParameter(name="params", type="string", description="Query parameters (JSON string)", required=False),
        ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=30),
    ]
    permissions = [ToolPermission.NETWORK]
    tags = ["api", "http", "rest", "request"]
    timeout_seconds = 60.0

    def _execute(self, url: str, method: str = "GET", headers: str = None,
                 body: str = None, params: str = None, timeout: int = 30) -> ToolResult:

        try:
            import requests
        except ImportError:
            return ToolResult(
                success=False,
                error="requests library not installed. Run: pip install requests",
            )

        # Parse headers
        req_headers = {"User-Agent": "NICTO-AI/1.0"}
        if headers:
            try:
                req_headers.update(json.loads(headers))
            except json.JSONDecodeError:
                return ToolResult(success=False, error="Invalid headers JSON")

        # Parse body
        req_body = None
        if body:
            try:
                req_body = json.loads(body)
            except json.JSONDecodeError:
                req_body = body  # Send as raw string

        # Parse params
        req_params = None
        if params:
            try:
                req_params = json.loads(params)
            except json.JSONDecodeError:
                return ToolResult(success=False, error="Invalid params JSON")

        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=req_headers,
                json=req_body if isinstance(req_body, dict) else None,
                data=req_body if isinstance(req_body, str) else None,
                params=req_params,
                timeout=timeout,
            )

            # Parse response
            try:
                response_json = response.json()
                output = response_json
            except json.JSONDecodeError:
                output = response.text[:5000]

            return ToolResult(
                success=response.status_code < 400,
                output={
                    "status_code": response.status_code,
                    "status_text": response.reason,
                    "headers": dict(response.headers),
                    "body": output,
                    "url": response.url,
                },
                error=f"HTTP {response.status_code}: {response.reason}" if response.status_code >= 400 else None,
            )

        except requests.Timeout:
            return ToolResult(success=False, error=f"Request timed out after {timeout}s")
        except requests.ConnectionError:
            return ToolResult(success=False, error=f"Could not connect to {url}")
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {str(e)}")
