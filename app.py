import streamlit as st
import importlib.util
import pickle
from datetime import datetime
from intent_agent_adaptive_v8_FINAL import (
    IntentAgentSession,
    extract_structured_intent_evidence,
    apply_intent_policy,
    fuse_intent_evidence,
)

st.set_page_config(
    page_title="Agentic AI Payment Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .stApp { background:#f7fbff; color:#142033; }
    .block-container { max-width:1180px; padding:0.55rem 0.8rem 0.8rem !important; }
    html, body, [class*="css"] { font-size:13px; color:#142033; }
    h1 { font-size:1.35rem !important; margin:.1rem 0 .2rem !important; color:#10233f !important; }
    h2,h3 { font-size:1.02rem !important; margin:.3rem 0 !important; color:#10233f !important; }
    h4 { font-size:.92rem !important; margin:.25rem 0 !important; color:#10233f !important; }
    p, label, span, div { line-height:1.25; }
    div[data-testid="stVerticalBlock"] { gap:.28rem !important; }
    div[data-testid="stMetric"] {
        background:#ffffff; border:1px solid #d8e4f0; padding:.32rem .48rem !important;
        border-radius:9px; box-shadow:0 2px 8px rgba(30,60,90,.05);
    }
    div[data-testid="stMetricLabel"] { color:#52657a !important; font-size:.70rem !important; }
    div[data-testid="stMetricValue"] { color:#10233f !important; font-size:1rem !important; }
    div[data-baseweb="input"] input, div[data-baseweb="base-input"] input,
    div[data-baseweb="select"] *, textarea {
        color:#111827 !important; -webkit-text-fill-color:#111827 !important;
    }
    div[data-baseweb="input"], div[data-baseweb="base-input"],
    div[data-baseweb="select"] > div { background:#ffffff !important; }
    button[data-baseweb="tab"] {
        color:#17324f !important; background:#eef5fb !important; font-size:.70rem !important;
        padding:.24rem .38rem !important; min-height:1.8rem !important; border-radius:6px 6px 0 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color:#0b3b69 !important; background:#ffffff !important; font-weight:700 !important;
    }
    div[data-baseweb="tab-list"] { gap:.10rem !important; }
    .stButton > button { min-height:2.1rem !important; padding:.25rem .55rem !important; font-size:.82rem !important; }
    .decision-box { border-radius:10px; padding:.55rem .75rem; margin:.25rem 0 .45rem; }
    .decision-label { color:#52657a; font-size:.66rem; font-weight:700; letter-spacing:.08em; }
    .decision-value { color:#10233f; font-size:1.22rem; font-weight:800; margin-top:.1rem; }
    .decision-source { color:#52657a; font-size:.68rem; margin-top:.08rem; }
    .decision-approve { background:#e8f7ed; border:2px solid #35a854; }
    .decision-review { background:#fff6d8; border:2px solid #d7a900; }
    .decision-decline { background:#fde8e8; border:2px solid #d84b4b; }
    .decision-neutral { background:#eef5fb; border:1px solid #b9cadb; }
    [class*="shap"], .shap-value, .shap-value * { font-size:.70rem !important; line-height:1.05 !important; }
    div[data-testid="stDataFrame"] { font-size:.70rem !important; }
    #MainMenu, footer { visibility:hidden; }
    header { background:transparent !important; }
    @media(max-width:768px){
        .block-container{padding:.35rem .45rem .55rem !important;}
        h1{font-size:1.12rem !important;} h2,h3{font-size:.92rem !important;}
        html,body,[class*="css"]{font-size:12px;}
        button[data-baseweb="tab"]{font-size:.62rem !important;padding:.18rem .25rem !important;}
    }
    
/* V3: prevent result cards from overlapping */
.decision-box {
    position: relative !important;
    min-height: 64px !important;
    height: auto !important;
    box-sizing: border-box !important;
    margin: .35rem 0 .85rem !important;
    overflow: visible !important;
}
div[data-testid="stMetric"] {
    min-height: 58px !important;
    height: auto !important;
    box-sizing: border-box !important;
    overflow: visible !important;
    margin-bottom: .18rem !important;
}
div[data-testid="stMetricLabel"],
div[data-testid="stMetricValue"] {
    white-space: normal !important;
    overflow: visible !important;
}
div[data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
    row-gap: .45rem !important;
    margin-bottom: .35rem !important;
}
div[data-baseweb="tab-list"] {
    position: relative !important;
    clear: both !important;
    margin-top: .45rem !important;
    padding-top: .15rem !important;
}
div[data-testid="stMetric"] > div {
    height: auto !important;
    min-height: 0 !important;
}
@media(max-width:768px) {
    .decision-box { min-height:58px !important; margin-bottom:.7rem !important; }
    div[data-testid="stMetric"] { min-height:54px !important; }
}


/* V4: stronger decision colours and compact fraud split */
.decision-approve { background:#9BE3AE !important; border:2px solid #238B45 !important; }
.decision-review  { background:#FFE07A !important; border:2px solid #C58B00 !important; }
.decision-decline { background:#FF9B9B !important; border:2px solid #C93636 !important; }
.decision-neutral { background:#CFE5F7 !important; border:2px solid #6C9FC5 !important; }

button[data-baseweb="tab"] {
    background:#DDEAF5 !important;
    color:#10233F !important;
    border:1px solid #B8CBDC !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background:#8EC5F0 !important;
    color:#071A2C !important;
    border:1px solid #4E91C5 !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-panel"] {
    padding-top:.35rem !important;
}

</style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD PLATFORM
# ============================================================

@st.cache_resource(show_spinner="Initializing payment intelligence engines...")
def load_platform():
    spec = importlib.util.spec_from_file_location(
        "payment_recovery",
        "payment_platform_FINAL_RECOVERY_LOCAL_GEMINI.py",
    )

    payment_recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(payment_recovery)

    with open("payment_platform_LIGHT_DEPLOYMENT_STATE.pkl", "rb") as f:
        checkpoint = pickle.load(f)

    payment_recovery.restore_runtime_state(checkpoint)
    payment_recovery.recovery = payment_recovery

    return payment_recovery


payment_recovery = load_platform()

# ============================================================
# SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "transaction_payload" not in st.session_state:
    st.session_state.transaction_payload = None

if "intent_session" not in st.session_state:
    st.session_state.intent_session = None

if "intent_overlay" not in st.session_state:
    st.session_state.intent_overlay = None

if "intent_current_question" not in st.session_state:
    st.session_state.intent_current_question = None

if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "SETUP"

# ============================================================
# HELPERS
# ============================================================

def pretty(value):
    if value is None:
        return "—"
    return str(value).replace("_", " ").title()


def reset_for_new_transaction():
    """Reset workflow state so another transaction can be analyzed."""
    st.session_state.analysis_result = None
    st.session_state.transaction_payload = None
    st.session_state.intent_session = None
    st.session_state.intent_overlay = None
    st.session_state.intent_current_question = None
    st.session_state.workflow_stage = "SETUP"
    st.session_state.setup_screen = 1

    for key in list(st.session_state.keys()):
        if key.startswith("intent_answer_"):
            del st.session_state[key]


def render_decision(decision, source):
    normalized = str(decision or "").upper()
    if "APPROV" in normalized or "ALLOW" in normalized:
        decision_class = "decision-approve"
    elif "DECLIN" in normalized or "BLOCK" in normalized or "REJECT" in normalized:
        decision_class = "decision-decline"
    elif "REVIEW" in normalized or "HOLD" in normalized or "STEP" in normalized:
        decision_class = "decision-review"
    else:
        decision_class = "decision-neutral"

    st.markdown(
        f"""
        <div class="decision-box {decision_class}">
            <div class="decision-label">FINAL PLATFORM DECISION</div>
            <div class="decision-value">{pretty(decision)}</div>
            <div class="decision-source">Decision source · {pretty(source)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# HEADER
# ============================================================

st.caption("PAYMENT INTELLIGENCE")
st.title("Agentic AI Payment Platform")
st.caption("Fraud · Policy · Authentication · Currency · Agentic Intent")

# ============================================================
# INPUT AREA
# ============================================================

if st.session_state.workflow_stage == "SETUP":
    st.subheader("Transaction Setup")
    st.caption("Complete the transaction details in two compact screens.")

    if "setup_screen" not in st.session_state:
        st.session_state.setup_screen = 1

    # Persistent draft values across the two setup screens.
    defaults = {
        "draft_customer_id": 1,
        "draft_merchant_id": 1,
        "draft_amount": 1000.0,
        "draft_country": None,
        "draft_channel": "CNP",
        "draft_device_id": None,
        "draft_entry_mode": "ECOM",
        "draft_date": datetime.now().date(),
        "draft_time": datetime.now().replace(second=0, microsecond=0).time(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    countries = sorted(payment_recovery.df["country"].dropna().unique().tolist())

    if st.session_state.setup_screen == 1:
        st.markdown("### 1 of 2 · Core transaction")

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            customer_id = st.number_input(
                "Customer ID (1–5000)",
                min_value=1, max_value=5000,
                value=int(st.session_state.draft_customer_id), step=1,
            )
        with r1c2:
            merchant_id = st.number_input(
                "Merchant ID (1–5000)",
                min_value=1, max_value=5000,
                value=int(st.session_state.draft_merchant_id), step=1,
            )

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            amount = st.number_input(
                "Transaction Amount (₹)",
                min_value=0.01,
                value=float(st.session_state.draft_amount),
                step=100.00,
                format="%.2f",
            )
        with r2c2:
            default_country = (
                st.session_state.draft_country
                if st.session_state.draft_country in countries
                else ("India" if "India" in countries else countries[0])
            )
            country = st.selectbox(
                "Transaction Country",
                countries,
                index=countries.index(default_country),
            )

        st.caption("Next: device, channel, entry mode and transaction time.")

        if st.button("Next →", type="primary", use_container_width=True):
            st.session_state.draft_customer_id = int(customer_id)
            st.session_state.draft_merchant_id = int(merchant_id)
            st.session_state.draft_amount = float(amount)
            st.session_state.draft_country = country

            customer_history = payment_recovery.df[
                payment_recovery.df["customer_id"] == int(customer_id)
            ]
            customer_devices = sorted(
                int(x) for x in customer_history["device_id"].dropna().unique().tolist()
            )
            st.session_state.draft_device_id = customer_devices[0] if customer_devices else 1
            st.session_state.setup_screen = 2
            st.rerun()

        st.stop()

    # SCREEN 2
    customer_id = int(st.session_state.draft_customer_id)
    merchant_id = int(st.session_state.draft_merchant_id)
    amount = float(st.session_state.draft_amount)
    country = st.session_state.draft_country

    customer_history = payment_recovery.df[
        payment_recovery.df["customer_id"] == customer_id
    ]
    customer_devices = sorted(
        int(x) for x in customer_history["device_id"].dropna().unique().tolist()
    )

    st.markdown("### 2 of 2 · Payment context")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        if customer_devices:
            current_device = (
                st.session_state.draft_device_id
                if st.session_state.draft_device_id in customer_devices
                else customer_devices[0]
            )
            device_id = st.selectbox(
                "Recognized Device",
                customer_devices,
                index=customer_devices.index(current_device),
            )
        else:
            device_id = st.number_input(
                "Device ID",
                min_value=1,
                value=int(st.session_state.draft_device_id or 1),
                step=1,
            )

    with r1c2:
        channel_options = ["CNP", "POS", "CONTACTLESS"]
        channel = st.selectbox(
            "Payment Channel",
            channel_options,
            index=channel_options.index(
                st.session_state.draft_channel
                if st.session_state.draft_channel in channel_options else "CNP"
            ),
        )

    entry_mode_map = {
        "CNP": ["ECOM", "MANUAL"],
        "POS": ["CHIP", "SWIPE"],
        "CONTACTLESS": ["TAP"],
    }

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        modes = entry_mode_map[channel]
        entry_mode = st.selectbox(
            "Entry Mode",
            modes,
            index=modes.index(
                st.session_state.draft_entry_mode
                if st.session_state.draft_entry_mode in modes else modes[0]
            ),
        )
    with r2c2:
        transaction_date = st.date_input(
            "Transaction Date",
            value=st.session_state.draft_date,
        )

    transaction_time = st.time_input(
        "Transaction Time",
        value=st.session_state.draft_time,
    )

    transaction_datetime = datetime.combine(
        transaction_date, transaction_time
    ).strftime("%Y-%m-%d %H:%M:%S")

    nav1, nav2 = st.columns([1, 2])
    with nav1:
        if st.button("← Back", use_container_width=True):
            st.session_state.setup_screen = 1
            st.rerun()

    with nav2:
        analyze = st.button(
            "Analyze Transaction →",
            type="primary",
            use_container_width=True,
        )

# ============================================================
# ANALYSIS
# ============================================================

analyze = locals().get("analyze", False)

if analyze:
    payload = {
        "customer_id": int(customer_id),
        "amount": float(amount),
        "merchant_id": int(merchant_id),
        "country": country,
        "channel": channel,
        "entry_mode": entry_mode,
        "device_id": int(device_id),
        "transaction_datetime": transaction_datetime,
    }

    try:
        with st.spinner("Running fraud, policy, authentication and currency intelligence..."):
            result = payment_recovery.analyze_payment_intelligence(**payload)

        st.session_state.analysis_result = result
        st.session_state.transaction_payload = payload
        st.session_state.intent_overlay = None
        st.session_state.intent_current_question = None

        fraud = result.get("fraud_intelligence") or {}
        risk_decision = fraud.get("risk_decision") or {}
        model_risk_score = float(risk_decision.get("model_risk_score", 0.0))

        customer_avg = float(customer_history["amount"].mean()) if (
            not customer_history.empty and "amount" in customer_history.columns
        ) else 0.0

        amount_ratio = (
            float(amount) / customer_avg
            if customer_avg > 0
            else 0.0
        )

        intent_session = IntentAgentSession(
            risk_score=model_risk_score,
            amount=float(amount),
            amount_vs_customer_avg=amount_ratio,
            transaction_context={
                "channel": channel,
                "country": country,
                "entry_mode": entry_mode,
            },
        )

        # Agentic Intent is triggered by the EXISTING PLATFORM POLICY layer.
        # The IntentAgentSession still receives the model/transaction context for
        # question generation and evidence assessment, but no longer owns the trigger.
        review_policy = result.get("transaction_review_policy") or {}
        policy_decision = result.get("policy_decision") or {}
        triggered_rules = review_policy.get("triggered_rules") or []

        policy_action = str(
            policy_decision.get("final_decision")
            or review_policy.get("final_decision")
            or review_policy.get("decision")
            or ""
        ).upper()

        policy_triggered = bool(triggered_rules) or any(
            token in policy_action
            for token in ("REVIEW", "HOLD", "DECLINE", "BLOCK", "REJECT")
        )

        st.session_state.intent_session = (
            intent_session if policy_triggered else None
        )

        if policy_triggered:
            with st.spinner("Preparing adaptive payment-intent verification..."):
                st.session_state.intent_current_question = intent_session.next_question()
            st.session_state.workflow_stage = "VERIFY"
        else:
            st.session_state.workflow_stage = "RESULTS"

        st.rerun()

    except Exception as error:
        st.error("Transaction analysis could not be completed.")
        with st.expander("Technical details"):
            st.exception(error)

result = st.session_state.analysis_result

# ============================================================
# AGENTIC INTENT VERIFICATION — V8 FROZEN BACKEND
# ============================================================

intent_session = st.session_state.intent_session

if result and intent_session is not None and st.session_state.intent_overlay is None:
    st.caption("SCREEN 3 OF 3 · PAYMENT VERIFICATION")
    st.subheader("Security check")
    st.caption("Up to 3 short questions · one question at a time")

    answered = len(intent_session.qa_pairs)
    st.progress(answered / 3)
    st.caption(f"Question {answered + 1} of 3")

    question = st.session_state.intent_current_question
    if question:
        st.markdown(f"### {question}")
        answer = st.text_input(
            "Your answer",
            key=f"intent_answer_{answered + 1}",
            placeholder="Type your answer...",
        )

        back_col, continue_col = st.columns([1, 2])
        with back_col:
            if st.button("← Back", use_container_width=True, key=f"intent_back_{answered + 1}"):
                st.session_state.analysis_result = None
                st.session_state.intent_session = None
                st.session_state.intent_overlay = None
                st.session_state.intent_current_question = None
                st.session_state.setup_screen = 2
                st.session_state.workflow_stage = "SETUP"
                st.rerun()

        with continue_col:
            continue_verification = st.button(
                "Continue verification →",
                type="primary",
                use_container_width=True,
                key=f"intent_submit_{answered + 1}",
            )

        if continue_verification:
            if not answer.strip():
                st.warning("Please enter an answer before continuing.")
            else:
                try:
                    # V8 session marks itself complete as soon as Q3 has been
                    # generated. Therefore Q3's answer must be recorded directly
                    # instead of calling submit_answer(), whose guard correctly
                    # blocks calls once complete=True.
                    if len(intent_session.qa_pairs) == 2:
                        intent_session.qa_pairs.append({
                            "question_number": 3,
                            "question": question,
                            "customer_answer": answer.strip(),
                        })

                        with st.spinner("Evaluating payment-intent evidence..."):
                            extracted = extract_structured_intent_evidence(
                                intent_session.qa_pairs
                            )
                            policy_result = apply_intent_policy(extracted)
                            fusion = fuse_intent_evidence(
                                intent_session.risk_score,
                                policy_result,
                            )

                        st.session_state.intent_overlay = {
                            "structured_evidence": extracted,
                            "deterministic_intent_policy": policy_result,
                            "decision_fusion": fusion,
                        }
                        st.session_state.intent_current_question = None
                        st.session_state.workflow_stage = "RESULTS"

                    else:
                        intent_session.submit_answer(answer)
                        with st.spinner("Preparing the next adaptive question..."):
                            st.session_state.intent_current_question = (
                                intent_session.next_question()
                            )

                    st.rerun()

                except Exception as error:
                    st.error("Intent verification could not be completed.")
                    with st.expander("Technical details"):
                        st.exception(error)

    st.stop()


# ============================================================
# RESULTS
# ============================================================

if result and st.session_state.workflow_stage == "RESULTS":
    title_col, action_col = st.columns([3, 1])
    with title_col:
        st.subheader("Decision Command Center")
        st.caption("Consolidated model, policy, authentication and currency intelligence.")
    with action_col:
        if st.button("↻ New Transaction", type="primary", use_container_width=True, key="new_transaction_top"):
            reset_for_new_transaction()
            st.rerun()

    final_decision = result.get("platform_decision", "UNKNOWN")
    decision_source = result.get("decision_source", "UNKNOWN")

    render_decision(final_decision, decision_source)

    fraud = result.get("fraud_intelligence") or {}
    risk_decision = fraud.get("risk_decision") or {}
    policy = result.get("policy_decision") or {}
    auth = result.get("step_up_authentication") or {}
    currency = result.get("currency_intelligence") or {}
    merchant = result.get("merchant_control") or {}

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric("Model Risk Score", f"{risk_decision.get('model_risk_score', 0):.3f}")
    k2.metric("ML Decision", pretty(risk_decision.get("decision")))
    k3.metric("Policy Decision", pretty(policy.get("final_decision")))
    k4.metric("Merchant", pretty(merchant.get("merchant_status")))
    k5.metric("Recommended Currency", currency.get("recommended_currency", "—"))

    copilot_tab, model_tab, intent_tab, policy_tab, auth_tab, currency_tab, executive_tab = st.tabs(
        [
            "Copilot Recommendation",
            "Model Evidence",
            "Agentic Intent",
            "Policy & Merchant",
            "Authentication",
            "Currency Intelligence",
            "Executive Summary",
        ]
    )

    copilot = fraud.get("investigator_copilot") or {}
    explainability = fraud.get("explainability") or {}
    risk_indicators = explainability.get("risk_indicators") or []
    reducing_indicators = explainability.get("risk_reducing_indicators") or []

    with copilot_tab:
        st.markdown("#### Investigator Copilot Recommendation")
        st.caption(f"Source · {copilot.get('source', '—')}")
        st.write(
            copilot.get(
                "investigator_explanation",
                "No investigator narrative available.",
            )
        )

    with model_tab:
        a, b = st.columns(2)
        a.metric(
            "Model Risk Score",
            f"{risk_decision.get('model_risk_score', 0):.4f}",
        )
        b.metric(
            "Model Decision",
            pretty(risk_decision.get("decision")),
        )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Risk Drivers")
            if risk_indicators:
                for item in risk_indicators[:4]:
                    c1, c2 = st.columns([3, 1])
                    c1.write(
                        f"**{item.get('indicator', '—')}** · "
                        f"{item.get('value', '—')}"
                    )
                    c2.caption(
                        f"SHAP {item.get('shap_contribution', 0):.3f}"
                        if isinstance(item.get("shap_contribution"), (int, float))
                        else f"SHAP {item.get('shap_contribution', 0)}"
                    )
            else:
                st.caption("No material risk-increasing indicators.")

        with right:
            st.markdown("#### Risk-Reducing Signals")
            if reducing_indicators:
                for item in reducing_indicators[:4]:
                    c1, c2 = st.columns([3, 1])
                    c1.write(
                        f"**{item.get('indicator', '—')}** · "
                        f"{item.get('value', '—')}"
                    )
                    c2.caption(
                        f"SHAP {item.get('shap_contribution', 0):.3f}"
                        if isinstance(item.get("shap_contribution"), (int, float))
                        else f"SHAP {item.get('shap_contribution', 0)}"
                    )
            else:
                st.caption("No material risk-reducing indicators.")

    with intent_tab:
        st.subheader("Agentic Payment Intent Layer")

        overlay = st.session_state.intent_overlay

        if overlay:
            policy_intent = overlay.get("deterministic_intent_policy") or {}
            fusion_intent = overlay.get("decision_fusion") or {}
            extracted = overlay.get("structured_evidence") or {}

            i1, i2, i3 = st.columns(3)
            i1.metric("Intent Policy Score", policy_intent.get("policy_score", 0))
            i2.metric("Intent Risk Band", pretty(policy_intent.get("policy_band")))
            i3.metric(
                "Intent Overlay",
                pretty(fusion_intent.get("intent_overlay_action")),
            )

            st.markdown("#### Customer Intent Evidence")

            evidence_items = [
                ("Purpose", extracted.get("purpose")),
                ("Contact Channel", extracted.get("contact_channel")),
                ("Discount", f"{extracted.get('discount_percent'):g}%" if extracted.get("discount_percent") is not None else None),
                ("Claimed Relationship", extracted.get("claimed_relationship")),
                ("Cross-Border", "Yes" if extracted.get("cross_border") else "No"),
                ("Origin", extracted.get("origin_country")),
                ("Destination", extracted.get("destination_country")),
                ("Delivery Mode", extracted.get("delivery_mode")),
                ("Third-Party Direction", "Detected" if extracted.get("third_party_direction") else None),
                ("Urgency / Pressure", "Detected" if extracted.get("urgency_or_pressure") else None),
                ("Secrecy / Coaching", "Detected" if extracted.get("secrecy_or_coaching") else None),
                ("Outside Standard Checkout", "Detected" if extracted.get("payment_outside_standard_checkout") else None),
            ]

            visible_evidence = [
                (label, value) for label, value in evidence_items
                if value not in (None, "", "—")
            ]

            for row_start in range(0, len(visible_evidence), 3):
                row = visible_evidence[row_start:row_start + 3]
                cols = st.columns(len(row))
                for col, (label, value) in zip(cols, row):
                    with col:
                        st.caption(label)
                        st.markdown(f"**{value}**")

            st.markdown("#### Intent Policy Signals")
            signals = policy_intent.get("policy_signals") or []
            if signals:
                for signal in signals:
                    s1, s2, s3 = st.columns([2.2, 2, 0.8])
                    s1.write(pretty(signal.get("signal")))
                    s2.write(signal.get("observed", "—"))
                    s3.metric("Weight", signal.get("weight", 0))
            else:
                st.success("No deterministic adverse intent signals were extracted.")

            st.caption(
                "Gemini asks adaptive questions and extracts structured claims. "
                "Deterministic Python policy assigns the intent-risk signals; "
                "the existing platform remains the owner of the final payment decision."
            )
        else:
            st.info(
                "The Agentic Intent layer was not triggered for this transaction."
            )

    with policy_tab:
        st.subheader("Merchant & Transaction Policy")

        p1, p2, p3 = st.columns(3)
        p1.metric("Merchant Status", pretty(merchant.get("merchant_status")))
        p2.metric("Policy Decision", pretty(policy.get("final_decision")))
        p3.metric("Decision Source", pretty(policy.get("decision_source")))

        review_policy = result.get("transaction_review_policy") or {}
        triggered_rules = review_policy.get("triggered_rules") or []

        st.markdown("#### Triggered Rules")
        if triggered_rules:
            for rule in triggered_rules:
                with st.container(border=True):
                    st.markdown(f"**{pretty(rule.get('reason'))}**")
                    st.caption(
                        f"Rule: {pretty(rule.get('rule'))} | "
                        f"Observed: {rule.get('observed_value', '—')} | "
                        f"Threshold: {rule.get('threshold', '—')} | "
                        f"Action: {pretty(rule.get('action'))}"
                    )
        else:
            st.success("No transaction review policy rules were triggered.")

        with st.expander("Merchant control details"):
            if isinstance(merchant, dict):
                for key, value in merchant.items():
                    st.write(f"**{pretty(key)}:** {value}")
            else:
                st.write(merchant)

    with auth_tab:
        st.subheader("Step-Up Authentication")

        a1, a2, a3 = st.columns(3)
        a1.metric("Pre-Authentication", pretty(auth.get("pre_auth_decision")))
        a2.metric("Step-Up Required", "YES" if auth.get("step_up_required") else "NO")
        a3.metric("Final Status", pretty(auth.get("final_decision")))

        if auth.get("step_up_required") and auth.get("authentication_outcome") is None:
            st.warning(
                "This transaction requires additional customer authentication before a final decision."
            )

            try:
                outcomes = list(payment_recovery.step_up_auth_policy["outcomes"].keys())
            except Exception:
                outcomes = []

            if outcomes:
                authentication_outcome = st.selectbox(
                    "Authentication Outcome",
                    outcomes,
                )

                if st.button(
                    "Submit Authentication Result",
                    type="primary",
                    key="submit_auth",
                ):
                    payload = dict(st.session_state.transaction_payload)
                    payload["authentication_outcome"] = authentication_outcome

                    try:
                        with st.spinner("Resolving authenticated payment decision..."):
                            updated_result = payment_recovery.analyze_payment_intelligence(**payload)

                        st.session_state.analysis_result = updated_result
                        st.rerun()

                    except Exception as error:
                        st.exception(error)
        else:
            if auth.get("authentication_outcome"):
                st.success(
                    f"Authentication outcome: {pretty(auth.get('authentication_outcome'))}"
                )

    with currency_tab:
        st.subheader("Currency Intelligence")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Recommended Currency", currency.get("recommended_currency", "—"))
        c2.metric("Customer Currency", currency.get("customer_home_currency", "—"))
        c3.metric("Merchant Currency", currency.get("merchant_home_currency", "—"))
        c4.metric("Estimated Cost", f"₹{currency.get('estimated_total_cost_inr', 0):,.2f}")

        options = currency.get("currency_options") or []

        if options:
            st.markdown("#### Currency Cost Comparison")

            for option in options:
                with st.container(border=True):
                    x1, x2, x3, x4 = st.columns(4)
                    x1.metric("Currency", option.get("currency", "—"))
                    x2.metric("Total Markup", f"{option.get('total_markup_percent', 0):.3f}%")
                    x3.metric("Estimated Cost", f"₹{option.get('estimated_total_cost_inr', 0):,.2f}")
                    x4.metric("Rank", f"#{option.get('cost_rank', '—')}")

        st.markdown("#### Currency Intelligence Assistant")
        with st.container(border=True):
            st.markdown(
                currency.get(
                    "assistant_explanation",
                    "No currency explanation available.",
                )
            )

    with executive_tab:
        st.subheader("Executive Decision Summary")

        render_decision(final_decision, decision_source)

        e1, e2 = st.columns(2, gap="large")

        with e1:
            st.metric("ML Fraud Decision", pretty(risk_decision.get("decision")))
            st.metric("Model Risk Score", f"{risk_decision.get('model_risk_score', 0):.6f}")
            st.metric("Policy Decision", pretty(policy.get("final_decision")))

        with e2:
            st.metric("Authentication", pretty(auth.get("final_decision")))
            st.metric("Merchant Status", pretty(merchant.get("merchant_status")))
            st.metric("Recommended Currency", currency.get("recommended_currency", "—"))

        decision_reasons = policy.get("decision_reason", [])
        if isinstance(decision_reasons, str):
            decision_reasons = [decision_reasons]

        if decision_reasons:
            st.markdown("#### Why the platform made this decision")
            for reason in decision_reasons:
                st.info(pretty(reason))

        st.markdown("#### Executive Summary")
        copilot_summary = (fraud.get("investigator_copilot") or {}).get(
            "investigator_explanation",
            "No investigator narrative available."
        )
        st.write(
            f"The existing fraud model assessed this transaction as "
            f"**{pretty(risk_decision.get('decision'))}** with a model risk score of "
            f"**{risk_decision.get('model_risk_score', 0):.3f}**. "
            f"The platform's final decision is **{pretty(final_decision)}**."
        )
        st.markdown("#### Investigator Copilot")
        st.write(copilot_summary)


    st.markdown("")
    if st.button("↻ Analyze Another Transaction", use_container_width=True, key="new_transaction_bottom"):
        reset_for_new_transaction()
        st.rerun()
