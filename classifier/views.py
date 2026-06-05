from django.shortcuts import render

# classifier/views.py

import json
import re

from rest_framework.decorators import api_view
from rest_framework.response import Response

from .gemini_classifier import classify_query


@api_view(["POST"])
def classify_message(request):

    message = request.data.get("message")

    result = classify_query(message)

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