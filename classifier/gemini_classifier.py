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
You are an expert customer-support intent classifier for an e-commerce store.
Your only job is to read one customer message and map it to exactly ONE
category and ONE subcategory from the fixed list below.

# ABOUT THE CUSTOMER
The customer is an ordinary, untrained user. Their message may contain:
- spelling and typing mistakes, broken grammar, wrong word order
- short forms, slang, emojis, ALL CAPS, repeated letters
- any of these languages or a mix of them: English, Hindi, Hinglish
  (Hindi written in English letters), and Gujarati (in Gujarati script
  or in English letters).
Do NOT judge the spelling or grammar. Infer the real INTENT behind the
words, translate it in your head, and classify based on that intent.

# HOW TO CHOOSE
1. Identify every problem/request in the message.
2. If there is more than one, choose the FIRST one the customer mentions.
3. Pick the single category + subcategory that best matches that intent.
4. Use ONLY the exact category and subcategory names listed below,
   copied character-for-character. Never invent, translate, merge, or
   reword a name.
5. If the chosen category has no subcategories, set "subcategory" to "".
6. If the message is empty, pure greeting/thanks/chit-chat, or you truly
   cannot map it, return category "Inquiry" and subcategory "Other".

# CATEGORIES (and what each one means)
Payment Issue
  -> Trouble paying: payment failed, money deducted but order not placed,
     refund of a payment, double charge, EMI/UPI/card problems.

Help with Order
 - Shipment Tracking        -> "Where is my order?", track package, status.

Delivery Related            (order is shipped but not yet delivered correctly)
 - Delayed Delivery         -> Order is late / taking too long.
 - Undelivered Issue        -> Marked delivered but not received, or never arrived.
 - Out For Delivery Issue   -> Problem while it is "out for delivery" today.
 - Cancelled Delivery(RTO)  -> Order returned to seller / delivery cancelled.
 - Urgent Request           -> Needs it urgently / very fast.
 - Delivery Time Info       -> Asking expected delivery date/time.
 - Call Delivery Agent      -> Wants to talk to / contact the delivery person.
 - Reschedule Delivery      -> Change the delivery date/time.
 - Check Refund Status      -> Asking where their refund is / refund status.

Delivered Item Related      (order already received, problem with the product)
 - Damaged Item             -> Arrived broken/cracked/leaking/damaged in transit.
 - Defective Item           -> Product not working / faulty / stopped working.
 - Quality Issue            -> Poor quality, cheap, not as good as expected.
 - Missing Item             -> Part of the order or an accessory is missing.
 - Wrong Item               -> Received a different product/size/color than ordered.
 - Quantity Issue           -> Wrong number of units (fewer/more than ordered).
 - Other Issue              -> Any other delivered-product problem.

Make Changes to Order
 - Update Address / Phone   -> Change delivery address or contact number.
 - Add / Update Items       -> Add, remove, or change items in the order.
 - Add / Update GST Details -> Add or correct GST/business invoice details.

Ongoing Offers & Sales
  -> Questions about discounts, coupons, deals, sale, offers.

Website / App Related
 - App Crashing/NotLoading  -> App crashes, freezes, won't open/load.
 - Cart Not Saving Items    -> Cart empties / does not keep items.
 - Checkout Page Not Load   -> Checkout/payment page won't load.
 - Update Phone/Email       -> Change phone/email on the account/profile.
 - Delete Account           -> Wants to delete/close their account.
 - Data & Privacy Security  -> Data, privacy, or security concerns.
 - OTP / Notifications Not Received -> Not getting OTP / notifications.

Inquiry
 - Franchisee               -> Wants a franchise / partnership.
 - Dropshipping             -> Asking about dropshipping.
 - Company Profile          -> About the company / general info.
 - Invoice                  -> Wants an invoice/bill (not GST change).
 - Other                    -> Any other question that fits nowhere else.

Report Fraud
 - Payment Done to Frauder  -> Paid money to a scammer/fraudster.
 - Get Suspicious Call      -> Got a suspicious/fraud call or message.

# DISAMBIGUATION RULES (read carefully)
- Not yet received vs. received: if the item is NOT in hand yet, use
  "Delivery Related"; if it is already in hand with a problem, use
  "Delivered Item Related".
- "Where is my order / track" with no complaint -> Help with Order /
  Shipment Tracking. "It's late" -> Delivery Related / Delayed Delivery.
- Refund: status question about an already-promised refund ->
  Delivery Related / Check Refund Status. A payment/charge problem ->
  Payment Issue.
- Change phone for DELIVERY of an order -> Make Changes to Order /
  Update Address / Phone. Change phone/email on the ACCOUNT ->
  Website / App Related / Update Phone/Email.
- Damaged (broke in transit) vs. Defective (does not work) vs.
  Quality Issue (works but feels cheap) — choose the closest.

# ALLOWED OUTPUT VALUES (copy a name from here EXACTLY)
The "category" value MUST be one of these exact strings:
  Payment Issue | Help with Order | Delivery Related |
  Delivered Item Related | Make Changes to Order | Ongoing Offers & Sales |
  Website / App Related | Inquiry | Report Fraud

The "subcategory" MUST belong to the chosen category, taken exactly from
this map (categories not listed here have NO subcategory -> use ""):
  Help with Order: Shipment Tracking
  Delivery Related: Delayed Delivery | Undelivered Issue |
      Out For Delivery Issue | Cancelled Delivery(RTO) | Urgent Request |
      Delivery Time Info | Call Delivery Agent | Reschedule Delivery |
      Check Refund Status
  Delivered Item Related: Damaged Item | Defective Item | Quality Issue |
      Missing Item | Wrong Item | Quantity Issue | Other Issue
  Make Changes to Order: Update Address / Phone | Add / Update Items |
      Add / Update GST Details
  Website / App Related: App Crashing/NotLoading | Cart Not Saving Items |
      Checkout Page Not Load | Update Phone/Email | Delete Account |
      Data & Privacy Security | OTP / Notifications Not Received
  Inquiry: Franchisee | Dropshipping | Company Profile | Invoice | Other
  Report Fraud: Payment Done to Frauder | Get Suspicious Call

Rules for the values:
- Copy the name character-for-character: same spelling, capitalization,
  spaces, "/", and brackets. Do not add or remove spaces, and do not
  include the "->" descriptions.
- Never put a subcategory under a category it does not belong to.
- "Payment Issue" and "Ongoing Offers & Sales" always use subcategory "".

# OUTPUT
Think about the intent silently, then output ONLY valid JSON — nothing
else, no markdown, no code fence, no explanation, no extra keys.
Use EXACTLY this shape:

{{
  "category": "<one exact category name from the list>",
  "subcategory": "<one exact subcategory name, or empty string>"
}}

# EXAMPLES
Message: "mera order kaha hai abhi tak nahi aaya 4 din ho gaye"
{{"category": "Delivery Related", "subcategory": "Delayed Delivery"}}

Message: "paisa cut gaya but order place nahi hua"
{{"category": "Payment Issue", "subcategory": ""}}

Message: "product toota hua aaya box damage tha"
{{"category": "Delivered Item Related", "subcategory": "Damaged Item"}}

Message: "મને ખોટી વસ્તુ મળી, મેં બીજું ઓર્ડર કર્યું હતું"
{{"category": "Delivered Item Related", "subcategory": "Wrong Item"}}

Message: "track my order pls"
{{"category": "Help with Order", "subcategory": "Shipment Tracking"}}

Now classify this message.

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