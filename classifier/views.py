from django.shortcuts import render

# classifier/views.py

import json
import re

from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .gemini_classifier import RateLimitError, classify_query


@api_view(["POST"])
def classify_message(request):

    message = request.data.get("message")

    if not message:
        return Response(
            {"error": "Field 'message' is required."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        result = classify_query(message)
    except RateLimitError:
        return Response(
            {"error": "Please try sometime later."},
            status=status.HTTP_429_TOO_MANY_REQUESTS,
        )
    except Exception as exc:
        # Never leak a raw 500 traceback to the client.
        return Response(
            {"error": "AI service is temporarily unavailable.", "detail": str(exc)},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        # Fall back to extracting the first {...} block from the raw text.
        match = re.search(r"\{.*\}", result or "", re.DOTALL)
        parsed = None
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    return Response(_normalize(parsed))


def _normalize(parsed):
    """Coerce Gemini's output into a consistent multi-issue shape:

        {"issue_count": <int>, "issues": [{"category": ..., "subcategory": ...}]}

    Handles the new array format, the legacy single-object format, and any
    malformed/empty response.
    """
    unknown = {"category": "Unknown", "subcategory": "Unknown"}

    # New format: already has an "issues" list.
    if isinstance(parsed, dict) and isinstance(parsed.get("issues"), list):
        issues = [_clean_issue(i) for i in parsed["issues"] if isinstance(i, dict)]
        return {"issue_count": len(issues), "issues": issues}

    # Legacy single-object format: {"category": ..., "subcategory": ...}
    if isinstance(parsed, dict) and "category" in parsed:
        issue = _clean_issue(parsed)
        return {"issue_count": 1, "issues": [issue]}

    # A bare list of issues.
    if isinstance(parsed, list):
        issues = [_clean_issue(i) for i in parsed if isinstance(i, dict)]
        return {"issue_count": len(issues), "issues": issues}

    # Anything else (None / unparseable): report a single Unknown issue.
    return {"issue_count": 1, "issues": [unknown]}


def _clean_issue(issue):
    return {
        "category": (issue.get("category") or "Unknown").strip() or "Unknown",
        "subcategory": (issue.get("subcategory") or "").strip(),
    }