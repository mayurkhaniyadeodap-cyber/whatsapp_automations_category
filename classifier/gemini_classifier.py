# classifier/gemini_classifier.py

import os
import time

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# This project's API key has no free-tier quota on gemini-2.0-flash
# (returns 429 with "limit: 0"), but gemini-2.5-flash works fine.
# Override via the GEMINI_MODEL env var if needed.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

model = genai.GenerativeModel(MODEL_NAME)


class RateLimitError(Exception):
    """Raised when Gemini's quota/rate limit is exhausted."""

def classify_query(user_message):

    prompt = f"""
You are a customer-support classifier for an e-commerce store.

The customer is an ordinary user, NOT trained. Their message may contain
spelling mistakes, broken grammar, wrong sentence structure, short forms,
slang, or mixed Hindi/English (Hinglish). Read carefully, understand the
real INTENT behind the words, and ignore the spelling/grammar errors.

Choose ONLY ONE category and ONE subcategory — the single MAIN issue.
If the message mentions more than one problem, pick the FIRST problem
mentioned in the message. If the chosen category has no subcategories,
use "". Pick only from the list below; never invent new categories or
subcategories.

Categories:

Payment Issue

Help with Order
- Shipment Tracking  

Delivery Related
- Delayed Delivery 
- Undelivered Issue 
- Out For Delivery Issue 
- Cancelled Delivery(RTO) 
- Urgent Request 
- Delivery Time Info
- Call Delivery Agent 
- Reschedule Delivery 

Delivered Item Related
- Damaged Item
- Defective Item 
- Quality Issue
- Missing Item 
- Wrong Item 
- Quantity Issue 
- Other Issue

Make Changes to Order
- Update Address / Phone 
- Add / Update Items 
- Add / Update GST Details 

Ongoing Offers & Sales 

Website / App Related
- App Crashing/NotLoading 
- Cart Not Saving Items
- Checkout Page Not Load
- Update Phone/Email
- Delete Account
- Data & Privacy Security
- OTP / Notifications Not Received


Inquiry
- Franchisee
- Dropshipping
- Company Profile
- Invoice
- Other
    
Report Fraud
- Payment Done to Frauder
- Get Suspicious Call

Return ONLY valid JSON in EXACTLY this format (no extra text, no markdown):

{{
  "category": "",
  "subcategory": ""
}}

Customer Message:
{user_message}
"""

    # Retry a couple of times on transient rate limits before giving up.
    last_error = None
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
            break
        except ResourceExhausted as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 * (attempt + 1))  # 2s, then 4s
    else:
        raise RateLimitError(str(last_error))

    text = response.text.strip()

    # Gemini often wraps JSON in a markdown code fence (```json ... ```).
    # Strip it so the caller can json.loads() the result.
    if text.startswith("```"):
        text = text.strip("`")
        # remove an optional leading language hint like "json"
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]

    return text.strip()