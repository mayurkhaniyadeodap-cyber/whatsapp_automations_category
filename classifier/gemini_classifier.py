# classifier/gemini_classifier.py

import os

import google.generativeai as genai

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

def classify_query(user_message):

    prompt = f"""
You are a support classifier.

Choose ONLY one category and subcategory.

Categories:

Payment Issue

Help with Order
- Shipment Tracking
- Delivery Related

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

Offer / Discount Related

Website / App Related
- App Crashing
- Cart Not Saving Items
- Saved Address Not Found
- Browser & Device Support
- Checkout Page Not Load

Account Related
- Password Reset
- Update Phone/Email
- Delete Account
- Data & Privacy Security
- OTP / Notifications Not Received
- View Order History
- Create New Account

Return JSON:

{{
 "category":"",
 "subcategory":""
}}

Customer Message:
{user_message}             
"""

    response = model.generate_content(prompt)

    text = response.text.strip()

    # Gemini often wraps JSON in a markdown code fence (```json ... ```).
    # Strip it so the caller can json.loads() the result.
    if text.startswith("```"):
        text = text.strip("`")
        # remove an optional leading language hint like "json"
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]

    return text.strip()