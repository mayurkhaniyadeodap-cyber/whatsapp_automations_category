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
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = {"category": "Unknown", "subcategory": "Unknown"}
        else:
            parsed = {"category": "Unknown", "subcategory": "Unknown"}

    return Response(parsed)