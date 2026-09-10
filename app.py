import streamlit as st
import importlib.util
import pickle
from datetime import datetime

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Payment Intelligence Platform",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# LIGHT ENTERPRISE PAYMENT THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 5% 0%, rgba(0,120,215,0.08) 0%, transparent 28%),
            radial-gradient(circle at 95% 3%, rgba(255,145,70,0.05) 0%, transparent 22%),
            linear-gradient(135deg, #fbfdff 0%, #f2f8ff 48%, #ffffff 100%) !important;
        color: #172033 !important;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .stApp p,
    .stApp span,
    .stApp label {
        color: #334155 !important;
    }

    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: #102a43 !important;
        letter-spacing: -0.02em;
    }

    /* INPUT LABELS */
    label[data-testid="stWidgetLabel"] p {
        color: #1f2937 !important;
        font-weight: 650 !important;
    }

    /* NUMBER / TEXT INPUT BOXES */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: rgba(37,99,235,0.18) !important;
    }

    input {
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* NUMBER INPUT STEPPER AREA */
    button[aria-label="Step up"],
    button[aria-label="Step down"] {
        color: #1f2937 !important;
    }

    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.96) !important;
        border: 1px solid rgba(37,99,235,0.11) !important;
        box-shadow: 0 6px 20px rgba(31,73,125,0.06) !important;
        border-radius: 14px !important;
        padding: 18px 20px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748b !important;
    }

    div[data-testid="stMetricValue"] {
        color: #102a43 !important;
    }

    /* TABS */
    button[data-baseweb="tab"] {
        color: #52657a !important;
        font-weight: 600 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1261a6 !important;
        font-weight: 700 !important;
    }

    /* BUTTONS */
    .stButton > button {
        border-radius: 10px !important;
        min-height: 46px !important;
        font-weight: 700 !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #1261a6, #1683c7) !important;
        border: 1px solid #1261a6 !important;
        box-shadow: 0 5px 14px rgba(18,97,166,0.16) !important;
    }

    /* FORCE PRIMARY BUTTON TEXT TO WHITE */
    .stButton > button[kind="primary"],
    .stButton > button[kind="primary"] *,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 750 !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #0f5593, #1376b6) !important;
    }

    /* DECISION CARD */
    .decision-box {
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(18,97,166,0.14);
        border-left: 5px solid #1683c7;
        border-radius: 16px;
        padding: 20px 22px;
        margin: 8px 0 18px 0;
        box-shadow: 0 6px 20px rgba(31,73,125,0.05);
    }

    .decision-label {
        color: #64748b !important;
        font-size: 11px;
        letter-spacing: .12em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .decision-value {
        color: #102a43 !important;
        font-size: 28px;
        font-weight: 760;
        margin-top: 4px;
    }

    .decision-source {
        color: #718096 !important;
        font-size: 12px;
        margin-top: 4px;
    }

    .screen-kicker {
        color: #1261a6 !important;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: .10em;
        margin-bottom: 4px;
    }

    .timestamp-note {
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(37,99,235,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        color: #475569 !important;
        font-size: 13px;
    }

    hr {
        border-color: rgba(30,64,110,0.10) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}
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
        "payment_platform_FINAL_RECOVERY.txt.py",
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

if "page" not in st.session_state:
    st.session_state.page = "input"

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "transaction_payload" not in st.session_state:
    st.session_state.transaction_payload = None


# ============================================================
# HELPERS
# ============================================================

def pretty(value):
    if value is None:
        return "—"
    return str(value).replace("_", " ").title()


def header():
    st.caption("PAYMENT RISK | DECISIONING | INTELLIGENCE")
    st.title("AI Payment Intelligence Platform")


def render_decision(decision, source):
    st.markdown(
        f"""
        <div class="decision-box">
            <div class="decision-label">Final Platform Decision</div>
            <div class="decision-value">{pretty(decision)}</div>
            <div class="decision-source">Decision source · {pretty(source)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SCREEN 1 — TRANSACTION SETUP
# ============================================================

if st.session_state.page == "input":

    header()

    st.write(
        "Real-time transaction risk assessment, behavioral intelligence, "
        "policy orchestration, explainability, authentication and currency decisioning."
    )

    st.success("ALL INTELLIGENCE ENGINES OPERATIONAL")

    st.markdown("---")
    st.markdown(
        '<div class="screen-kicker">STEP 01 · TRANSACTION SETUP</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Create Transaction")

    st.caption(
        "Enter the transaction context. Historical customer and device behavior "
        "is retrieved automatically."
    )

    left, right = st.columns(2, gap="large")

    with left:

        customer_id = st.number_input(
            "Customer ID",
            min_value=1,
            max_value=5000,
            value=1,
            step=1,
            help="Valid range: 1 to 5000",
        )

        merchant_id = st.number_input(
            "Merchant Number",
            min_value=1,
            max_value=5000,
            value=1,
            step=1,
            help="Valid range: 1 to 5000",
        )

        amount = st.number_input(
            "Transaction Amount (INR)",
            min_value=0.01,
            value=1000.00,
            step=100.00,
            format="%.2f",
        )

    customer_history = payment_recovery.df[
        payment_recovery.df["customer_id"] == int(customer_id)
    ]

    customer_devices = sorted(
        int(x)
        for x in customer_history["device_id"].dropna().unique().tolist()
    )

    with right:

        if customer_devices:
            device_id = st.selectbox(
                "Recognized Device",
                customer_devices,
                help="Devices historically observed for the selected customer.",
            )
        else:
            device_id = st.number_input(
                "Device ID",
                min_value=1,
                value=1,
                step=1,
            )

        countries = sorted(
            payment_recovery.df["country"].dropna().unique().tolist()
        )

        country = st.selectbox(
            "Transaction Country",
            countries,
        )

        channel = st.selectbox(
            "Payment Channel",
            ["CNP", "POS", "CONTACTLESS"],
        )

    entry_mode_map = {
        "CNP": ["ECOM", "MANUAL"],
        "POS": ["CHIP", "SWIPE"],
        "CONTACTLESS": ["TAP"],
    }

    mode_col, time_col = st.columns(2, gap="large")

    with mode_col:
        entry_mode = st.selectbox(
            "Entry Mode",
            entry_mode_map[channel],
        )

    with time_col:
        st.markdown(
            """
            <div class="timestamp-note">
                <strong>Transaction Timestamp</strong><br>
                Captured automatically from the system when
                <strong>Analyze Transaction</strong> is clicked.
                Past or future timestamps cannot be entered manually.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Customer History",
        f"{len(customer_history):,} transactions",
    )
    m2.metric(
        "Known Devices",
        len(customer_devices),
    )
    m3.metric(
        "Selected Device",
        device_id,
    )

    st.markdown("")

    if st.button(
        "Analyze Transaction",
        type="primary",
        use_container_width=True,
    ):

        # Capture current system date/time only at submission.
        transaction_datetime = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

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

            with st.spinner(
                "Running fraud, policy, authentication and currency intelligence..."
            ):

                result = (
                    payment_recovery
                    .analyze_payment_intelligence(
                        **payload
                    )
                )

            st.session_state.analysis_result = result
            st.session_state.transaction_payload = payload
            st.session_state.page = "results"
            st.rerun()

        except Exception as error:

            st.error(
                "Transaction analysis could not be completed."
            )

            with st.expander(
                "Technical details"
            ):
                st.exception(
                    error
                )


# ============================================================
# SCREEN 2 — DECISION COMMAND CENTER
# ============================================================

elif st.session_state.page == "results":

    result = st.session_state.analysis_result

    header()

    top_left, top_right = st.columns(
        [4, 1]
    )

    with top_left:

        st.markdown(
            '<div class="screen-kicker">STEP 02 · DECISION COMMAND CENTER</div>',
            unsafe_allow_html=True,
        )

        st.subheader(
            "Transaction Intelligence Result"
        )

    with top_right:

        if st.button(
            "← New Transaction",
            use_container_width=True,
        ):
            st.session_state.analysis_result = None
            st.session_state.transaction_payload = None
            st.session_state.page = "input"
            st.rerun()

    if not result:
        st.warning(
            "No transaction result is available."
        )
        st.stop()

    final_decision = result.get(
        "platform_decision",
        "UNKNOWN",
    )

    decision_source = result.get(
        "decision_source",
        "UNKNOWN",
    )

    render_decision(
        final_decision,
        decision_source,
    )

    fraud = (
        result.get(
            "fraud_intelligence"
        )
        or {}
    )

    risk_decision = (
        fraud.get(
            "risk_decision"
        )
        or {}
    )

    policy = (
        result.get(
            "policy_decision"
        )
        or {}
    )

    auth = (
        result.get(
            "step_up_authentication"
        )
        or {}
    )

    currency = (
        result.get(
            "currency_intelligence"
        )
        or {}
    )

    merchant = (
        result.get(
            "merchant_control"
        )
        or {}
    )

    # --------------------------------------------------------
    # ECONOMIC DECISIONING
    # --------------------------------------------------------
    transaction_amount = float(
        (result.get("transaction") or {}).get(
            "amount",
            (st.session_state.transaction_payload or {}).get("amount", 0)
        )
    )

    try:
        false_decline_economics = (
            payment_recovery.calculate_false_decline_cost(
                transaction_amount
            )
        )
    except Exception:
        false_decline_economics = {}

    try:
        fraud_loss_economics = (
            payment_recovery.calculate_fraud_loss(
                transaction_amount
            )
        )
    except Exception:
        fraud_loss_economics = {}

    false_decline_cost = float(
        false_decline_economics.get(
            "total_false_decline_cost_inr",
            0
        )
    )

    fraud_loss_exposure = float(
        fraud_loss_economics.get(
            "fraud_loss_inr",
            0
        )
    )

    if fraud_loss_exposure > false_decline_cost:
        economic_tradeoff = "Fraud Loss Dominates"
        economic_explanation = (
            "The modeled fraud-loss exposure is greater than the estimated "
            "cost of falsely declining a legitimate transaction. This supports "
            "stronger risk controls when the fraud and policy layers identify "
            "material risk."
        )
    elif false_decline_cost > fraud_loss_exposure:
        economic_tradeoff = "Customer Friction Cost Dominates"
        economic_explanation = (
            "The estimated false-decline cost is greater than the modeled "
            "fraud-loss exposure. This highlights the commercial cost of "
            "unnecessarily declining a legitimate transaction."
        )
    else:
        economic_tradeoff = "Costs Balanced"
        economic_explanation = (
            "The modeled fraud-loss exposure and false-decline cost are "
            "approximately equal for this transaction."
        )

    k1, k2, k3, k4, k5 = st.columns(5)

    k1.metric(
        "Model Risk Score",
        f"{risk_decision.get('model_risk_score', 0):.3f}",
    )

    k2.metric(
        "ML Decision",
        pretty(
            risk_decision.get(
                "decision"
            )
        ),
    )

    k3.metric(
        "Policy Decision",
        pretty(
            policy.get(
                "final_decision"
            )
        ),
    )

    k4.metric(
        "Merchant",
        pretty(
            merchant.get(
                "merchant_status"
            )
        ),
    )

    k5.metric(
        "Recommended Currency",
        currency.get(
            "recommended_currency",
            "—",
        ),
    )

    st.markdown("")

    (
        fraud_tab,
        policy_tab,
        auth_tab,
        economic_tab,
        currency_tab,
        executive_tab,
    ) = st.tabs(
        [
            "Fraud Intelligence",
            "Policy & Merchant",
            "Authentication",
            "Economic Decisioning",
            "Currency Intelligence",
            "Executive Summary",
        ]
    )

    # --------------------------------------------------------
    # FRAUD TAB
    # --------------------------------------------------------

    with fraud_tab:

        st.subheader(
            "Fraud & Behavioral Intelligence"
        )

        explainability = (
            fraud.get(
                "explainability"
            )
            or {}
        )

        risk_indicators = (
            explainability.get(
                "risk_indicators"
            )
            or []
        )

        reducing_indicators = (
            explainability.get(
                "risk_reducing_indicators"
            )
            or []
        )

        a, b = st.columns(2)

        a.metric(
            "Model Risk Score",
            f"{risk_decision.get('model_risk_score', 0):.6f}",
        )

        b.metric(
            "Original Model Decision",
            pretty(
                risk_decision.get(
                    "decision"
                )
            ),
        )

        st.markdown(
            "#### Risk Drivers"
        )

        if risk_indicators:

            for item in risk_indicators:

                with st.container(
                    border=True
                ):

                    x1, x2, x3 = st.columns(
                        [2.2, 1.2, 1]
                    )

                    x1.write(
                        item.get(
                            "indicator",
                            "—",
                        )
                    )

                    x2.write(
                        item.get(
                            "value",
                            "—",
                        )
                    )

                    x3.metric(
                        "SHAP",
                        item.get(
                            "shap_contribution",
                            0,
                        ),
                    )

        else:

            st.info(
                "No material risk-increasing indicators identified."
            )

        st.markdown(
            "#### Risk-Reducing Signals"
        )

        if reducing_indicators:

            for item in reducing_indicators:

                with st.container(
                    border=True
                ):

                    x1, x2, x3 = st.columns(
                        [2.2, 1.2, 1]
                    )

                    x1.write(
                        item.get(
                            "indicator",
                            "—",
                        )
                    )

                    x2.write(
                        item.get(
                            "value",
                            "—",
                        )
                    )

                    x3.metric(
                        "SHAP",
                        item.get(
                            "shap_contribution",
                            0,
                        ),
                    )

        else:

            st.info(
                "No material risk-reducing indicators identified."
            )

        copilot = (
            fraud.get(
                "investigator_copilot"
            )
            or {}
        )

        st.markdown(
            "#### Investigator Copilot"
        )

        with st.container(
            border=True
        ):

            st.caption(
                f"Source: {copilot.get('source', '—')}"
            )

            st.markdown(
                copilot.get(
                    "investigator_explanation",
                    "No investigator narrative available.",
                )
            )

    # --------------------------------------------------------
    # POLICY TAB
    # --------------------------------------------------------

    with policy_tab:

        st.subheader(
            "Merchant & Transaction Policy"
        )

        p1, p2, p3 = st.columns(3)

        p1.metric(
            "Merchant Status",
            pretty(
                merchant.get(
                    "merchant_status"
                )
            ),
        )

        p2.metric(
            "Policy Decision",
            pretty(
                policy.get(
                    "final_decision"
                )
            ),
        )

        p3.metric(
            "Decision Source",
            pretty(
                policy.get(
                    "decision_source"
                )
            ),
        )

        review_policy = (
            result.get(
                "transaction_review_policy"
            )
            or {}
        )

        triggered_rules = (
            review_policy.get(
                "triggered_rules"
            )
            or []
        )

        st.markdown(
            "#### Triggered Rules"
        )

        if triggered_rules:

            for rule in triggered_rules:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**{pretty(rule.get('reason'))}**"
                    )

                    st.caption(
                        f"Rule: {pretty(rule.get('rule'))} | "
                        f"Observed: {rule.get('observed_value', '—')} | "
                        f"Threshold: {rule.get('threshold', '—')} | "
                        f"Action: {pretty(rule.get('action'))}"
                    )

        else:

            st.success(
                "No transaction review policy rules were triggered."
            )

        with st.expander(
            "Merchant control details"
        ):

            st.json(
                merchant
            )

    # --------------------------------------------------------
    # AUTHENTICATION TAB
    # --------------------------------------------------------

    with auth_tab:

        st.subheader(
            "Step-Up Authentication"
        )

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Pre-Authentication",
            pretty(
                auth.get(
                    "pre_auth_decision"
                )
            ),
        )

        a2.metric(
            "Step-Up Required",
            (
                "YES"
                if auth.get(
                    "step_up_required"
                )
                else "NO"
            ),
        )

        a3.metric(
            "Final Status",
            pretty(
                auth.get(
                    "final_decision"
                )
            ),
        )

        if (
            auth.get(
                "step_up_required"
            )
            and
            auth.get(
                "authentication_outcome"
            )
            is None
        ):

            st.warning(
                "This transaction requires additional customer authentication "
                "before a final decision."
            )

            try:

                outcomes = list(
                    payment_recovery
                    .step_up_auth_policy[
                        "outcomes"
                    ]
                    .keys()
                )

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

                    payload = dict(
                        st.session_state
                        .transaction_payload
                    )

                    payload[
                        "authentication_outcome"
                    ] = authentication_outcome

                    try:

                        with st.spinner(
                            "Resolving authenticated payment decision..."
                        ):

                            updated_result = (
                                payment_recovery
                                .analyze_payment_intelligence(
                                    **payload
                                )
                            )

                        st.session_state.analysis_result = updated_result
                        st.rerun()

                    except Exception as error:

                        st.exception(
                            error
                        )

        elif auth.get(
            "authentication_outcome"
        ):

            st.success(
                f"Authentication outcome: "
                f"{pretty(auth.get('authentication_outcome'))}"
            )

    # --------------------------------------------------------
    # ECONOMIC DECISIONING TAB
    # --------------------------------------------------------

    with economic_tab:

        st.subheader(
            "Economic Decisioning"
        )

        st.caption(
            "Commercial impact of fraud loss versus false-decline friction. "
            "This layer provides economic evidence supporting the platform "
            "decision; it does not independently override the fraud, policy "
            "or authentication decision."
        )

        ec1, ec2, ec3 = st.columns(3)

        ec1.metric(
            "Transaction Amount",
            f"₹{transaction_amount:,.2f}",
        )

        ec2.metric(
            "Fraud Loss Exposure",
            f"₹{fraud_loss_exposure:,.2f}",
        )

        ec3.metric(
            "False-Decline Cost",
            f"₹{false_decline_cost:,.2f}",
        )

        st.markdown(
            "#### Economic Trade-off"
        )

        with st.container(border=True):
            st.markdown(
                f"### {economic_tradeoff}"
            )
            st.write(
                economic_explanation
            )

        left_cost, right_cost = st.columns(
            2,
            gap="large",
        )

        with left_cost:

            st.markdown(
                "#### Cost of False Decline"
            )

            with st.container(border=True):

                st.metric(
                    "Customer Service / Operations",
                    f"₹{float(false_decline_economics.get('service_cost_inr', 0)):,.2f}",
                )

                st.metric(
                    "Lost Transaction Revenue",
                    f"₹{float(false_decline_economics.get('lost_transaction_revenue_inr', 0)):,.2f}",
                )

                st.metric(
                    "Expected Attrition / Friction",
                    f"₹{float(false_decline_economics.get('expected_attrition_cost_inr', 0)):,.2f}",
                )

                st.metric(
                    "Total False-Decline Cost",
                    f"₹{false_decline_cost:,.2f}",
                )

        with right_cost:

            st.markdown(
                "#### Cost of Fraud"
            )

            with st.container(border=True):

                loss_rate = float(
                    fraud_loss_economics.get(
                        "loss_given_fraud_rate",
                        0
                    )
                )

                st.metric(
                    "Transaction Value",
                    f"₹{transaction_amount:,.2f}",
                )

                st.metric(
                    "Loss Given Fraud",
                    f"{loss_rate * 100:.2f}%",
                )

                st.metric(
                    "Fraud Loss if Fraud Occurs",
                    f"₹{fraud_loss_exposure:,.2f}",
                )

        st.info(
            "Economic interpretation: the fraud-loss figure represents the "
            "modeled loss if fraud occurs, while false-decline cost represents "
            "the modeled commercial cost of wrongly declining a legitimate "
            "transaction. The platform's final action remains governed by the "
            "fraud model, policy controls and authentication workflow."
        )

    # --------------------------------------------------------
    # CURRENCY TAB
    # --------------------------------------------------------

    with currency_tab:

        st.subheader(
            "Currency Intelligence"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Recommended Currency",
            currency.get(
                "recommended_currency",
                "—",
            ),
        )

        c2.metric(
            "Customer Currency",
            currency.get(
                "customer_home_currency",
                "—",
            ),
        )

        c3.metric(
            "Merchant Currency",
            currency.get(
                "merchant_home_currency",
                "—",
            ),
        )

        c4.metric(
            "Estimated Cost",
            f"₹{currency.get('estimated_total_cost_inr', 0):,.2f}",
        )

        options = (
            currency.get(
                "currency_options"
            )
            or []
        )

        if options:

            st.markdown(
                "#### Currency Cost Comparison"
            )

            for option in options:

                with st.container(
                    border=True
                ):

                    x1, x2, x3, x4 = st.columns(
                        4
                    )

                    x1.metric(
                        "Currency",
                        option.get(
                            "currency",
                            "—",
                        ),
                    )

                    x2.metric(
                        "Total Markup",
                        f"{option.get('total_markup_percent', 0):.3f}%",
                    )

                    x3.metric(
                        "Estimated Cost",
                        f"₹{option.get('estimated_total_cost_inr', 0):,.2f}",
                    )

                    x4.metric(
                        "Rank",
                        f"#{option.get('cost_rank', '—')}",
                    )

        st.markdown(
            "#### Currency Intelligence Assistant"
        )

        with st.container(
            border=True
        ):

            st.markdown(
                currency.get(
                    "assistant_explanation",
                    "No currency explanation available.",
                )
            )

    # --------------------------------------------------------
    # EXECUTIVE SUMMARY TAB
    # --------------------------------------------------------

    with executive_tab:

        st.subheader(
            "Executive Decision Summary"
        )

        render_decision(
            final_decision,
            decision_source,
        )

        e1, e2 = st.columns(
            2,
            gap="large",
        )

        with e1:

            st.metric(
                "ML Fraud Decision",
                pretty(
                    risk_decision.get(
                        "decision"
                    )
                ),
            )

            st.metric(
                "Model Risk Score",
                f"{risk_decision.get('model_risk_score', 0):.6f}",
            )

            st.metric(
                "Policy Decision",
                pretty(
                    policy.get(
                        "final_decision"
                    )
                ),
            )

        with e2:

            st.metric(
                "Authentication",
                pretty(
                    auth.get(
                        "final_decision"
                    )
                ),
            )

            st.metric(
                "Merchant Status",
                pretty(
                    merchant.get(
                        "merchant_status"
                    )
                ),
            )

            st.metric(
                "Recommended Currency",
                currency.get(
                    "recommended_currency",
                    "—",
                ),
            )

        st.markdown(
            "#### Economic Decisioning Summary"
        )

        ee1, ee2, ee3 = st.columns(3)

        ee1.metric(
            "Fraud Loss Exposure",
            f"₹{fraud_loss_exposure:,.2f}",
        )

        ee2.metric(
            "False-Decline Cost",
            f"₹{false_decline_cost:,.2f}",
        )

        ee3.metric(
            "Economic Trade-off",
            economic_tradeoff,
        )

        with st.container(border=True):
            st.markdown(
                "**Economic Interpretation**"
            )
            st.write(
                economic_explanation
            )
            st.caption(
                "Economic evidence supports interpretation of the platform "
                "decision and does not independently replace fraud, policy "
                "or authentication controls."
            )

        decision_reasons = policy.get(
            "decision_reason",
            [],
        )

        if isinstance(
            decision_reasons,
            str,
        ):
            decision_reasons = [
                decision_reasons
            ]

        if decision_reasons:

            st.markdown(
                "#### Why the platform made this decision"
            )

            for reason in decision_reasons:

                st.info(
                    pretty(
                        reason
                    )
                )

        with st.expander(
            "View complete platform response"
        ):

            st.json(
                result
            )
