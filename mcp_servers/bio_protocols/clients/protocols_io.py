"""protocols.io public API client.

HTTP client for the protocols.io public API v3.
Handles:
  - Search for public protocols by query
  - Fetch individual protocol metadata
  - Rate limiting and error handling

Requires PROTOCOLS_IO_CLIENT_TOKEN environment variable (a valid access token).

Reference: SciCoreAgent 需求文档, Section 4.2.
  https://protocols.io/developers
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# protocols.io API base URL — NO www subdomain
PROTOCOLS_IO_API_BASE = "https://protocols.io/api/v3"

# Rate limit: 100 req/min per user → ~1.7 req/s, we use 1.5
MIN_INTERVAL = 0.67


class ProtocolsIOClient:
    """HTTP client for protocols.io public API."""

    def __init__(self, token: Optional[str] = None):
        self._token = token or os.environ.get("PROTOCOLS_IO_CLIENT_TOKEN", "")
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; SciCoreAgent/1.0)",
        })

    @property
    def ready(self) -> bool:
        return bool(self._token)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 20) -> dict:
        """Search protocols.io for public protocols matching a query.

        Uses the v3 API: GET /protocols?filter=public&key=<keyword>&page_size=N

        Returns a dict with 'results' (list) and 'searched_count' (int).
        """
        warnings: list[str] = []
        results: list[dict] = []

        if not self._token:
            return {
                "results": [], "searched_count": 0,
                "warnings": ["PROTOCOLS_IO_CLIENT_TOKEN not configured."],
            }

        try:
            params = {
                "filter": "public",
                "key": query,           # 'key' is the search keyword param (not 'q')
                "page_size": min(max_results, 50),
            }
            data = self._api_get("protocols", params)

            if isinstance(data, dict):
                items = data.get("items", data if not data.get("items") else [])
                # Normalize: items may be a list of protocol objects
                if isinstance(items, list):
                    for item in items[:max_results]:
                        # The item itself may be a protocol object, or have a 'protocol' key
                        protocol = item.get("protocol", item) if isinstance(item, dict) else item
                        if isinstance(protocol, dict):
                            results.append({
                                "external_id": str(protocol.get("id", "")),
                                "title": protocol.get("title", "") or "",
                                "url": protocol.get("uri", protocol.get("url", "")),
                                "authors": self._extract_authors(protocol),
                                "summary": protocol.get("description", protocol.get("summary")),
                                "tags": self._extract_tags(protocol),
                                "access": "public" if protocol.get("is_public", True) else "unknown",
                            })
                else:
                    warnings.append(f"Unexpected 'items' type: {type(items)}")
            elif isinstance(data, list):
                # Some API versions return a bare list
                for item in data[:max_results]:
                    if isinstance(item, dict):
                        results.append({
                            "external_id": str(item.get("id", "")),
                            "title": item.get("title", "") or "",
                            "url": item.get("uri", item.get("url", "")),
                            "authors": self._extract_authors(item),
                            "summary": item.get("description", item.get("summary")),
                            "tags": self._extract_tags(item),
                            "access": "public" if item.get("is_public", True) else "unknown",
                        })
            else:
                warnings.append(f"Unexpected API response format: {type(data)}")

        except requests.HTTPError as e:
            msg = f"HTTP {e.response.status_code}"
            try:
                body = e.response.json()
                msg += f": {body.get('error_message', body.get('status_text', str(body)))}"
            except Exception:
                pass
            logger.warning("protocols.io search failed for '%s': %s", query, msg)
            warnings.append(f"Search failed: {msg}")
        except Exception as e:
            logger.warning("protocols.io search failed for '%s': %s", query, e)
            warnings.append(f"Search failed: {e}")

        return {
            "results": results,
            "searched_count": len(results),
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch(self, external_id: str) -> dict:
        """Fetch a single protocol by its external ID.

        Uses: GET /protocols/{id}

        Returns a dict with 'protocol' and 'fetch_status'.
        """
        warnings: list[str] = []

        if not self._token:
            return {
                "protocol": None, "fetch_status": "skipped",
                "warnings": ["PROTOCOLS_IO_CLIENT_TOKEN not configured."],
            }

        try:
            data = self._api_get(f"protocols/{external_id}")
            protocol_data = data.get("protocol", data)

            if not isinstance(protocol_data, dict):
                return {
                    "protocol": None, "fetch_status": "failed",
                    "warnings": [f"Unexpected response type: {type(protocol_data)}"],
                }

            protocol = {
                "external_id": str(protocol_data.get("id", external_id)),
                "title": protocol_data.get("title", "") or "",
                "url": protocol_data.get("uri", protocol_data.get("url", "")),
                "authors": self._extract_authors(protocol_data),
                "summary": protocol_data.get("description", protocol_data.get("summary")),
                "materials": self._extract_materials(protocol_data),
                "steps": self._extract_steps(protocol_data),
                "tags": self._extract_tags(protocol_data),
                "access": "public" if protocol_data.get("is_public", True) else "unknown",
            }
            return {
                "protocol": protocol,
                "fetch_status": "success",
                "warnings": warnings,
            }

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return {
                    "protocol": None, "fetch_status": "skipped",
                    "warnings": [f"Protocol {external_id} not found (404)"],
                }
            if e.response.status_code == 403:
                return {
                    "protocol": None, "fetch_status": "skipped",
                    "warnings": [f"Protocol {external_id} access denied (403) — may be private"],
                }
            return {
                "protocol": None, "fetch_status": "failed",
                "warnings": [f"HTTP {e.response.status_code}"],
            }
        except Exception as e:
            logger.warning("Failed to fetch protocol %s: %s", external_id, e)
            return {
                "protocol": None, "fetch_status": "failed",
                "warnings": [f"Fetch failed: {e}"],
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_authors(self, data: dict) -> list[str]:
        """Extract author names from protocol data."""
        authors: list[str] = []
        for field in ["authors", "creators", "creator"]:
            val = data.get(field, [])
            if isinstance(val, list):
                for a in val:
                    if isinstance(a, dict):
                        name = a.get("name", a.get("full_name", ""))
                        if name:
                            authors.append(name)
                    elif isinstance(a, str):
                        authors.append(a)
                if authors:
                    break
        return authors

    def _extract_materials(self, data: dict) -> list[str]:
        """Extract materials list from protocol data."""
        materials: list[str] = []
        for field in ["materials", "components", "reagents"]:
            val = data.get(field, [])
            if isinstance(val, list):
                for m in val:
                    if isinstance(m, dict):
                        name = m.get("name", m.get("title", ""))
                        if name:
                            materials.append(name)
                    elif isinstance(m, str):
                        materials.append(m)
                if materials:
                    break
        return materials

    def _extract_steps(self, data: dict) -> list[dict]:
        """Extract step list from protocol data."""
        steps: list[dict] = []
        for field in ["steps", "procedure", "methods"]:
            val = data.get(field, [])
            if isinstance(val, list):
                for i, s in enumerate(val):
                    if isinstance(s, dict):
                        steps.append({
                            "step_number": i + 1,
                            "description": s.get("description", s.get("text", str(s))),
                            "expected_duration": s.get("duration", s.get("expected_duration")),
                            "notes": s.get("notes", s.get("note")),
                        })
                    elif isinstance(s, str):
                        steps.append({
                            "step_number": i + 1,
                            "description": s,
                        })
                if steps:
                    break
        return steps

    def _extract_tags(self, data: dict) -> list[str]:
        """Extract tags/keywords from protocol data."""
        tags: list[str] = []
        for field in ["tags", "keywords", "categories", "subjects"]:
            val = data.get(field, [])
            if isinstance(val, list):
                for t in val:
                    if isinstance(t, dict):
                        tag = t.get("name") or t.get("title") or t.get("tag", "")
                        if tag:
                            tags.append(tag)
                    elif isinstance(t, str):
                        tags.append(t)
                if tags:
                    break
        return tags

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _rate_limit(self) -> None:
        """Enforce rate limiting (100 req/min)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _api_get(self, path: str, params: dict | None = None) -> dict | list:
        """Make a GET request to the protocols.io API."""
        self._rate_limit()
        url = f"{PROTOCOLS_IO_API_BASE}/{path.lstrip('/')}"
        try:
            resp = self._session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON response from {url}")
