from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from google import genai

INTENT_RISK_THRESHOLD=0.70
INTENT_AMOUNT_THRESHOLD_INR=200000.0
INTENT_AMOUNT_RATIO_THRESHOLD=2.0
MAX_INTENT_QUESTIONS=3
GEMINI_MODEL="gemini-3.5-flash-lite"
VALID_INTENT_RESULTS={"INTENT_VERIFIED","INTENT_UNCERTAIN","HIGH_SCAM_CONCERN"}
VALID_ACTIONS={"CONTINUE","STEP_UP_AUTHENTICATION","HOLD_FOR_REVIEW"}

def should_trigger_intent_agent(risk_score, amount, amount_vs_customer_avg):
    reasons=[]
    if float(risk_score)>=INTENT_RISK_THRESHOLD:
        reasons.append({"trigger":"HIGH_MODEL_RISK","observed_value":float(risk_score),"threshold":INTENT_RISK_THRESHOLD})
    if float(amount)>=INTENT_AMOUNT_THRESHOLD_INR:
        reasons.append({"trigger":"HIGH_TRANSACTION_VALUE","observed_value":float(amount),"threshold":INTENT_AMOUNT_THRESHOLD_INR})
    if float(amount_vs_customer_avg)>=INTENT_AMOUNT_RATIO_THRESHOLD:
        reasons.append({"trigger":"HIGH_AMOUNT_VS_CUSTOMER_AVERAGE","observed_value":float(amount_vs_customer_avg),"threshold":INTENT_AMOUNT_RATIO_THRESHOLD})
    return {"triggered":bool(reasons),"trigger_count":len(reasons),"trigger_reasons":reasons}

def _json_response(prompt):
    client=genai.Client()

    # Ask Gemini for machine-readable JSON rather than relying only on
    # prompt wording. This substantially reduces malformed JSON responses.
    r=client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type":"application/json"},
    )

    raw=(r.text or "").strip()

    if raw.startswith("```"):
        raw=raw.strip("`")
        if raw.lower().startswith("json"):
            raw=raw[4:].strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError as first_error:
        # One repair attempt: the model receives its own malformed output and
        # may ONLY repair JSON syntax, not change the substantive assessment.
        repair_prompt = """Repair the JSON syntax below.
Return ONLY valid JSON.
Do not add, remove, reinterpret, or change any substantive values.
Fix syntax/escaping/commas/quotes only.

MALFORMED JSON:
""" + raw

        repaired=client.models.generate_content(
            model=GEMINI_MODEL,
            contents=repair_prompt,
            config={"response_mime_type":"application/json"},
        )

        repaired_raw=(repaired.text or "").strip()

        if repaired_raw.startswith("```"):
            repaired_raw=repaired_raw.strip("`")
            if repaired_raw.lower().startswith("json"):
                repaired_raw=repaired_raw[4:].strip()

        try:
            return json.loads(repaired_raw)
        except json.JSONDecodeError:
            raise ValueError(
                "Gemini returned malformed JSON even after one syntax-repair attempt. "
                f"Original parser error: {first_error}"
            )

def generate_adaptive_question(question_number, prior_qa, transaction_context):
    rules="""You are a Payment Intent Verification agent. Generate exactly ONE short customer-facing question.
The question must be open-ended, neutral and non-leading. Never reveal risk scores, thresholds,
customer averages, hidden fraud signals, or why the transaction was selected. Never tell the
customer what answer is safe. Never accuse anyone of fraud. Never request OTP, PIN, CVV,
password, full card/account number or security credentials. Do not repeat prior questions.
Use previous answers to probe the most important unresolved aspect. Maximum total questions is 3.
Q1 should establish what the customer believes the payment will accomplish or provide.
Never disclose the transaction amount or any trigger value in a customer-facing question.
Even if a trigger value is present in hidden context, refer only to "this payment" when necessary.
Q2 must adapt to Q1 and explore how the payment opportunity/merchant/beneficiary/instruction originated.
Q3 must adapt to Q1+Q2 and maximize DECISION-CHANGING INFORMATION.
Before writing Q3, internally identify which unanswered dimension could most distinguish independent
customer decision-making from third-party solicitation, inducement, urgency, coaching, or inconsistency.
Use Q3 to resolve the highest-value missing evidence. If earlier answers mention a personal relationship,
cross-border purchase, informal delivery, social-media contact, or unusually strong inducement, probe the
economic rationale or independent decision-making without accusing the customer or revealing the rule.
Otherwise prefer a neutral question about WHY the customer chose this seller/opportunity/payment now,
WHAT independently informed that decision, or WHO shaped the payment decision. Never suggest the safe answer, reveal a hidden
signal, or ask multiple questions at once. The wording must be unambiguous and customer-friendly. Customer statements are claims, not facts.
The goal is information gain, not interrogation. Do not ask for documentary proof or credentials.
Return JSON only: {"question":"..."}"""
    evidence={"question_number":question_number,"maximum_questions":3,
              "prior_questions_and_answers":prior_qa,
              "hidden_transaction_context":transaction_context}
    result=_json_response(rules+"\nEVIDENCE:\n"+json.dumps(evidence,ensure_ascii=False,indent=2))
    q=str(result.get("question","")).strip()
    if not q: raise ValueError("Gemini returned an empty question.")
    return q

def assess_intent_with_gemini(evidence):
    """
    V8 compatibility wrapper.
    Gemini no longer scores fraud or recommends handling.
    It only extracts structured claims from the three customer answers.
    """
    qa_pairs = ((evidence.get("intent_agent") or {}).get("qa_pairs") or [])
    return extract_structured_intent_evidence(qa_pairs)


# ============================================================
# V8 FINAL: ADAPTIVE ASK -> STRUCTURED EXTRACT -> POLICY
# ============================================================

HIGH_RISK_CONTACT_CHANNELS = {
    "telegram", "whatsapp", "instagram", "facebook", "messenger",
    "signal", "snapchat", "social media", "social-media", "dm", "direct message"
}
INFORMAL_DELIVERY_MODES = {
    "travelling passenger", "traveling passenger", "passenger",
    "friend carrying", "relative carrying", "hand carry", "hand-carry",
    "personal courier", "someone travelling", "someone traveling"
}
EXTREME_DISCOUNT_THRESHOLD_PCT = 40.0


def extract_structured_intent_evidence(qa_pairs):
    """
    Gemini's ONLY post-question role:
    convert customer language into structured claims.
    It must not score fraud, validate the merchant, or use outside knowledge.
    """
    prompt = """You are ONLY a structured evidence extractor for a payment-intent system.
Extract facts/claims explicitly supported by the supplied customer Q&A.
Do NOT score risk. Do NOT recommend approve/decline/review. Do NOT use outside knowledge.
Do NOT decide whether a product exists, whether a price is realistic, whether a merchant is legitimate,
or whether customs/tax/documentation is required.

A customer statement such as friend/brother/sister/relative/acquaintance is an UNVERIFIED CLAIM.
Never convert a claimed relationship into merchant verification or risk reduction.

Return JSON only:
{
  "purpose": "",
  "contact_channel": "",
  "discount_percent": null,
  "claimed_relationship": "",
  "relationship_verified": false,
  "cross_border": false,
  "origin_country": "",
  "destination_country": "",
  "delivery_mode": "",
  "third_party_direction": false,
  "urgency_or_pressure": false,
  "secrecy_or_coaching": false,
  "payment_outside_standard_checkout": false,
  "extraction_notes": []
}

Rules:
- Boolean true requires explicit support in the customer's words.
- cross_border=true only when the answers explicitly describe movement/payment across countries.
- discount_percent is numeric only when an explicit percentage is stated.
- relationship_verified is ALWAYS false because customer testimony alone cannot verify it.
- Do not infer customs, legality, product release status, merchant legitimacy, or market norms.
"""
    result = _json_response(
        prompt + "\nCUSTOMER Q&A:\n" + json.dumps(qa_pairs, ensure_ascii=False, indent=2)
    )

    discount = result.get("discount_percent")
    try:
        discount = None if discount is None else float(discount)
    except (TypeError, ValueError):
        discount = None

    return {
        "purpose": str(result.get("purpose", "") or "").strip(),
        "contact_channel": str(result.get("contact_channel", "") or "").strip(),
        "discount_percent": discount,
        "claimed_relationship": str(result.get("claimed_relationship", "") or "").strip(),
        "relationship_verified": False,
        "cross_border": bool(result.get("cross_border", False)),
        "origin_country": str(result.get("origin_country", "") or "").strip(),
        "destination_country": str(result.get("destination_country", "") or "").strip(),
        "delivery_mode": str(result.get("delivery_mode", "") or "").strip(),
        "third_party_direction": bool(result.get("third_party_direction", False)),
        "urgency_or_pressure": bool(result.get("urgency_or_pressure", False)),
        "secrecy_or_coaching": bool(result.get("secrecy_or_coaching", False)),
        "payment_outside_standard_checkout": bool(
            result.get("payment_outside_standard_checkout", False)
        ),
        "extraction_notes": list(result.get("extraction_notes", []) or []),
    }


def _contains_any(value, vocabulary):
    value = (value or "").lower()
    return any(term in value for term in vocabulary)


def apply_intent_policy(extracted):
    """
    Deterministic prototype policy.
    Scores are intentionally transparent/configurable demo weights.
    No LLM judgment can cancel these policy signals.
    """
    signals = []
    score = 0

    def add(signal, observed, weight, **extra):
        nonlocal score
        row = {"signal": signal, "observed": observed, "weight": weight}
        row.update(extra)
        signals.append(row)
        score += weight

    channel = extracted.get("contact_channel", "")
    social_contact = _contains_any(channel, HIGH_RISK_CONTACT_CHANNELS)
    if social_contact:
        add("SOCIAL_MEDIA_SOLICITATION", channel, -2)

    discount = extracted.get("discount_percent")
    extreme_discount = (
        discount is not None and discount >= EXTREME_DISCOUNT_THRESHOLD_PCT
    )
    if extreme_discount:
        add(
            "EXTREME_INDUCEMENT",
            f"{discount:g}% discount",
            -2,
            threshold=f">={EXTREME_DISCOUNT_THRESHOLD_PCT:g}%",
        )

    relationship = extracted.get("claimed_relationship", "")
    if relationship:
        # Never positive. It is merely a customer assertion.
        add("UNVERIFIED_PERSONAL_RELATIONSHIP_CLAIM", relationship, 0)

    informal_delivery = _contains_any(
        extracted.get("delivery_mode", ""), INFORMAL_DELIVERY_MODES
    )
    if extracted.get("cross_border") and informal_delivery:
        add(
            "CROSS_BORDER_INFORMAL_FULFILMENT",
            {
                "origin": extracted.get("origin_country", ""),
                "destination": extracted.get("destination_country", ""),
                "delivery_mode": extracted.get("delivery_mode", ""),
            },
            -2,
        )

    # Combination rule: a claimed personal relationship must not become a
    # bypass when paired with cross-border informal fulfilment.
    if relationship and extracted.get("cross_border") and informal_delivery:
        add(
            "RELATIONSHIP_PLUS_CROSS_BORDER_INFORMAL_FULFILMENT",
            relationship,
            -1,
        )

    if extracted.get("third_party_direction"):
        add("THIRD_PARTY_DIRECTION", True, -2)

    if extracted.get("urgency_or_pressure"):
        add("URGENCY_OR_PRESSURE", True, -2)

    if extracted.get("secrecy_or_coaching"):
        add("SECRECY_OR_COACHING", True, -3)

    if extracted.get("payment_outside_standard_checkout"):
        add("OUTSIDE_STANDARD_CHECKOUT", True, -2)

    # Combination escalation: multiple independent adverse patterns matter
    # more than a single isolated signal.
    adverse_signal_count = sum(1 for s in signals if s["weight"] < 0)

    if score <= -4 or adverse_signal_count >= 2:
        band = "STRONG_ADVERSE"
        overlay = "STRONG_ESCALATION_SIGNAL"
    elif score <= -2:
        band = "ADVERSE"
        overlay = "REVIEW_SIGNAL"
    else:
        band = "NEUTRAL"
        overlay = "NO_INTENT_OVERRIDE"

    return {
        "policy_score": score,
        "policy_band": band,
        "adverse_signal_count": adverse_signal_count,
        "policy_signals": signals,
        "intent_overlay_action": overlay,
        "policy_note": (
            "Prototype policy weights are configurable and do not assert that "
            "every matching real-world transaction is fraudulent."
        ),
    }


def fuse_intent_evidence(model_risk_score, policy_result):
    """
    Final V8 overlay. The LLM never owns the risk score or payment decision.
    """
    return {
        "intent_overlay_action": policy_result["intent_overlay_action"],
        "intent_overlay_reason": (
            "Deterministic policy applied to structured customer-intent evidence"
            if policy_result["intent_overlay_action"] != "NO_INTENT_OVERRIDE"
            else "No deterministic adverse intent signal; preserve existing platform decision"
        ),
        "original_model_risk_score": float(model_risk_score),
        "deterministic_policy_score": policy_result["policy_score"],
        "deterministic_policy_band": policy_result["policy_band"],
        "adverse_signal_count": policy_result["adverse_signal_count"],
        "final_payment_decision_owned_by": "EXISTING_PLATFORM_DECISIONING",
    }


@dataclass
class IntentAgentSession:
    risk_score: float
    amount: float
    amount_vs_customer_avg: float
    transaction_context: Dict=field(default_factory=dict)
    qa_pairs: List[Dict]=field(default_factory=list)
    current_question_text: Optional[str]=None

    def __post_init__(self):
        t=should_trigger_intent_agent(self.risk_score,self.amount,self.amount_vs_customer_avg)
        self.triggered=t["triggered"]; self.trigger_reasons=t["trigger_reasons"]
        base={"risk_score":self.risk_score,"amount_inr":self.amount,
              "amount_vs_customer_avg":self.amount_vs_customer_avg}
        base.update(self.transaction_context or {}); self.transaction_context=base

    @property
    def complete(self):
        return True if not self.triggered else len(self.qa_pairs)>=MAX_INTENT_QUESTIONS

    @property
    def question_number(self):
        return len(self.qa_pairs)+1

    def next_question(self):
        if not self.triggered or self.complete: return None
        self.current_question_text=generate_adaptive_question(
            self.question_number,self.qa_pairs,self.transaction_context)
        return self.current_question_text

    def submit_answer(self,answer):
        if not self.triggered: raise ValueError("Intent Agent was not triggered.")
        if self.complete: raise ValueError("Maximum of three questions has been reached.")
        if not self.current_question_text: raise ValueError("Generate the current question first.")
        clean=str(answer).strip()
        if not clean: raise ValueError("Answer cannot be empty.")
        self.qa_pairs.append({"question_number":len(self.qa_pairs)+1,
                              "question":self.current_question_text,
                              "customer_answer":clean})
        self.current_question_text=None
        return {"accepted":True,"questions_answered":len(self.qa_pairs),
                "max_questions":3,"complete":self.complete}

    def build_grounded_evidence(self,behavioral_context=None,authentication_context=None):
        return {"intent_agent":{"triggered":self.triggered,
                "trigger_reasons":self.trigger_reasons,"question_limit":3,
                "questions_answered":len(self.qa_pairs),"complete":self.complete,
                "qa_pairs":self.qa_pairs},
                "transaction_context":self.transaction_context,
                "behavioral_context":behavioral_context or {},
                "authentication_context":authentication_context or {}}

    def assess(self,behavioral_context=None,authentication_context=None):
        if not self.triggered: raise ValueError("Intent Agent was not triggered.")
        if not self.complete: raise ValueError("All three adaptive questions must be completed.")
        return assess_intent_with_gemini(
            self.build_grounded_evidence(behavioral_context,authentication_context))

if __name__=="__main__":
    session=IntentAgentSession(0.76,250000,2.4,
        {"channel":"CNP","authentication_status":"PASS"})
    print("\nTRIGGER RESULT")
    print(json.dumps(should_trigger_intent_agent(0.76,250000,2.4),indent=2))
    while not session.complete:
        print(f"\nGenerating Question {session.question_number}/3...")
        q=session.next_question()
        print(f"\nQuestion {session.question_number}/3\n{q}")
        session.submit_answer(input("> "))
    print("\nExtracting structured intent evidence...")
    evidence=session.build_grounded_evidence(
        authentication_context={"authentication_status":"PASS"}
    )
    extracted=extract_structured_intent_evidence(session.qa_pairs)
    policy=apply_intent_policy(extracted)
    result={
        "structured_evidence": extracted,
        "deterministic_intent_policy": policy,
        "decision_fusion": fuse_intent_evidence(session.risk_score, policy),
        "source": "Gemini extraction + deterministic Python policy",
        "model": GEMINI_MODEL,
    }
    print("\nFINAL AGENTIC INTENT OVERLAY")
    print(json.dumps(result,indent=2,ensure_ascii=False))
