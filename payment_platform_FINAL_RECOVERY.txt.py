"""
PAYMENT INTELLIGENCE PLATFORM — RECOVERY MODULE

Restores:
- Fraud behavioral feature engine
- XGBoost fraud scoring
- SHAP explainability
- Fraud investigator Copilot
- Multi-currency recommendation engine
- Merchant restriction control
- Combined payment intelligence orchestrator

The trained model/data/state are loaded separately from the
STEP518 state checkpoint.

Gemini credentials are intentionally NOT stored in either file.
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import shap
import json
import time
import copy


# Gemini is optional.
# The deterministic platform remains functional without it.

client = None
sandbox_explainer = None


REQUIRED_STATE_KEYS = [
    "df",
    "X_test",
    "X_test_encoded",
    "y_test",
    "model",
    "Review_Threshold",
    "Decline_Threshold",
    "action_policy",
    "safe_feature_map",
    "fx_to_inr",
    "country_default_currency",
    "customer_home_country",
    "customer_billing_currency",
    "customer_currency_df",
    "merchant_primary_country",
    "merchant_accepted_currencies",
    "merchant_currency_df",
    "fx_pair_markup",
    "merchant_currency_markup",
    "restricted_merchant_df",
    "restricted_merchant_registry"
]


def restore_runtime_state(checkpoint):
    """
    Inject checkpoint state into this module and rebuild SHAP explainer.
    """

    global sandbox_explainer

    missing_keys = [
        key
        for key in REQUIRED_STATE_KEYS
        if key not in checkpoint
    ]

    if missing_keys:
        raise KeyError(
            f"Checkpoint is missing required keys: {missing_keys}"
        )

    for key in REQUIRED_STATE_KEYS:
        globals()[key] = checkpoint[key]

    sandbox_explainer = shap.TreeExplainer(
        model
    )

    return {
        "dataset_shape": tuple(df.shape),
        "encoded_model_features": int(
            X_test_encoded.shape[1]
        ),
        "review_threshold": float(
            Review_Threshold
        ),
        "decline_threshold": float(
            Decline_Threshold
        ),
        "active_merchants": len(
            merchant_primary_country
        ),
        "restricted_merchants": len(
            restricted_merchant_registry
        )
    }


def initialize_gemini_from_colab_secret(
    secret_name="GEMINI_API_KEY"
):
    """
    Optional Gemini initialization.

    The rest of the platform works without Gemini because
    deterministic fallbacks remain available.
    """

    global client

    try:
        from google.colab import userdata
        from google import genai

        api_key = userdata.get(
            secret_name
        )

        if not api_key:
            client = None

            return {
                "status": "not_initialized",
                "reason": "API key not found"
            }

        client = genai.Client(
            api_key=api_key
        )

        return {
            "status": "initialized",
            "source": secret_name
        }

    except Exception as error:

        client = None

        return {
            "status": "not_initialized",
            "reason": str(error)
        }



def analyze_sandbox_transaction(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    transaction_datetime = pd.Timestamp(
        transaction_datetime
    )


    # --------------------------------------------------
    # CUSTOMER HISTORY BEFORE THIS TRANSACTION
    # --------------------------------------------------

    customer_history = df[
        (df["customer_id"] == customer_id)
        &
        (df["timestamp"] < transaction_datetime)
    ].copy()


    previous_customer_transactions = len(
        customer_history
    )


    # --------------------------------------------------
    # AMOUNT HISTORY
    # --------------------------------------------------

    if previous_customer_transactions > 0:

        customer_avg_amount_before_txn = (
            customer_history[
                "amount"
            ].mean()
        )

        amount_vs_customer_avg = (
            amount
            /
            customer_avg_amount_before_txn
        )

        amount_vs_customer_avg_filled = (
            amount_vs_customer_avg
        )

    else:

        customer_avg_amount_before_txn = np.nan

        amount_vs_customer_avg = np.nan

        amount_vs_customer_avg_filled = 1.0


    # --------------------------------------------------
    # DEVICE HISTORY
    # --------------------------------------------------

    previous_device_uses = int(
        (
            customer_history[
                "device_id"
            ]
            == device_id
        ).sum()
    )

    new_device = int(
        previous_customer_transactions > 0
        and
        previous_device_uses == 0
    )


    # --------------------------------------------------
    # MERCHANT HISTORY
    # --------------------------------------------------

    previous_merchant_uses = int(
        (
            customer_history[
                "merchant_id"
            ]
            == merchant_id
        ).sum()
    )

    new_merchant = int(
        previous_customer_transactions > 0
        and
        previous_merchant_uses == 0
    )


    # --------------------------------------------------
    # COUNTRY HISTORY
    # --------------------------------------------------

    previous_country_uses = int(
        (
            customer_history[
                "country"
            ]
            == country
        ).sum()
    )

    new_country = int(
        previous_customer_transactions > 0
        and
        previous_country_uses == 0
    )

    if previous_customer_transactions > 0:

        country_history_share = (
            previous_country_uses
            /
            previous_customer_transactions
        )

    else:

        country_history_share = 0.0


    # --------------------------------------------------
    # PREVIOUS TRANSACTION TIMING
    # --------------------------------------------------

    if previous_customer_transactions > 0:

        previous_transaction_time = (
            customer_history[
                "timestamp"
            ].max()
        )

        time_since_last_transaction_min = (
            (
                transaction_datetime
                -
                previous_transaction_time
            ).total_seconds()
            /
            60
        )

        time_since_last_transaction_filled = (
            time_since_last_transaction_min
        )

    else:

        previous_transaction_time = pd.NaT

        time_since_last_transaction_min = np.nan

        time_since_last_transaction_filled = (
            999999
        )


    # --------------------------------------------------
    # VELOCITY FEATURES
    # --------------------------------------------------

    velocity_10min = int(
        (
            customer_history[
                "timestamp"
            ]
            >=
            (
                transaction_datetime
                -
                pd.Timedelta(
                    minutes=10
                )
            )
        ).sum()
    )


    txn_count_last_1h = int(
        (
            customer_history[
                "timestamp"
            ]
            >=
            (
                transaction_datetime
                -
                pd.Timedelta(
                    hours=1
                )
            )
        ).sum()
    )


    spend_last_24h = float(
        customer_history.loc[
            customer_history[
                "timestamp"
            ]
            >=
            (
                transaction_datetime
                -
                pd.Timedelta(
                    hours=24
                )
            ),
            "amount"
        ].sum()
    )


    # --------------------------------------------------
    # MERCHANT MCC
    # --------------------------------------------------

    merchant_rows = df[
        df["merchant_id"]
        ==
        merchant_id
    ]


    if merchant_rows.empty:

        raise ValueError(
            f"Merchant ID {merchant_id} "
            "does not exist in historical "
            "merchant data."
        )


    mcc = int(
        merchant_rows[
            "mcc"
        ].iloc[0]
    )


    # --------------------------------------------------
    # BASIC TRANSACTION FLAGS
    # --------------------------------------------------

    transaction_hour = (
        transaction_datetime.hour
    )

    foreign_transaction = int(
        country != "INDIA"
    )

    unusual_transaction_hour = int(
        transaction_hour < 6
    )

    CNP_flag = int(
        channel == "CNP"
    )


    # --------------------------------------------------
    # RETURN FEATURE EVIDENCE
    # --------------------------------------------------

    return {

        "customer_id":
            int(customer_id),

        "amount":
            float(amount),

        "merchant_id":
            int(merchant_id),

        "mcc":
            int(mcc),

        "country":
            country,

        "channel":
            channel,

        "entry_mode":
            entry_mode,

        "device_id":
            int(device_id),

        "transaction_datetime":
            transaction_datetime,

        "transaction_hour":
            int(transaction_hour),

        "foreign_transaction":
            int(foreign_transaction),

        "unusual_transaction_hour":
            int(
                unusual_transaction_hour
            ),

        "CNP_flag":
            int(CNP_flag),

        "previous_customer_transactions":
            int(
                previous_customer_transactions
            ),

        "customer_avg_amount_before_txn":
            customer_avg_amount_before_txn,

        "amount_vs_customer_avg":
            amount_vs_customer_avg,

        "amount_vs_customer_avg_filled":
            float(
                amount_vs_customer_avg_filled
            ),

        "previous_device_uses":
            int(previous_device_uses),

        "new_device":
            int(new_device),

        "previous_merchant_uses":
            int(previous_merchant_uses),

        "new_merchant":
            int(new_merchant),

        "previous_country_uses":
            int(previous_country_uses),

        "new_country":
            int(new_country),

        "country_history_share":
            float(country_history_share),

        "previous_transaction_time":
            previous_transaction_time,

        "time_since_last_transaction_min":
            time_since_last_transaction_min,

        "time_since_last_transaction_filled":
            float(
                time_since_last_transaction_filled
            ),

        "velocity_10min":
            int(velocity_10min),

        "txn_count_last_1h":
            int(txn_count_last_1h),

        "spend_last_24h":
            float(spend_last_24h)
    }



def build_sandbox_model_features(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    behavior = analyze_sandbox_transaction(
        customer_id=customer_id,
        amount=amount,
        merchant_id=merchant_id,
        country=country,
        channel=channel,
        entry_mode=entry_mode,
        device_id=device_id,
        transaction_datetime=transaction_datetime
    )


    sandbox_model_features = pd.DataFrame(
        [{
            "amount":
                behavior["amount"],

            "mcc":
                behavior["mcc"],

            "country":
                behavior["country"],

            "channel":
                behavior["channel"],

            "entry_mode":
                behavior["entry_mode"],

            "transaction_hour":
                behavior["transaction_hour"],

            "foreign_transaction":
                behavior["foreign_transaction"],

            "unusual_transaction_hour":
                behavior["unusual_transaction_hour"],

            "time_since_last_transaction_filled":
                behavior[
                    "time_since_last_transaction_filled"
                ],

            "amount_vs_customer_avg_filled":
                behavior[
                    "amount_vs_customer_avg_filled"
                ],

            "previous_device_uses":
                behavior["previous_device_uses"],

            "new_device":
                behavior["new_device"],

            "txn_count_last_1h":
                behavior["txn_count_last_1h"],

            "velocity_10min":
                behavior["velocity_10min"],

            "spend_last_24h":
                behavior["spend_last_24h"],

            "previous_merchant_uses":
                behavior["previous_merchant_uses"],

            "new_merchant":
                behavior["new_merchant"],

            "CNP_flag":
                behavior["CNP_flag"],

            "new_country":
                behavior["new_country"],

            "country_history_share":
                behavior["country_history_share"]
        }]
    )


    return sandbox_model_features



def run_fraud_sandbox(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    # --------------------------------------------------
    # 1. BUILD EXACT 20 MODEL FEATURES
    # --------------------------------------------------

    sandbox_model_features = (
        build_sandbox_model_features(
            customer_id=customer_id,
            amount=amount,
            merchant_id=merchant_id,
            country=country,
            channel=channel,
            entry_mode=entry_mode,
            device_id=device_id,
            transaction_datetime=
                transaction_datetime
        )
    )


    # --------------------------------------------------
    # 2. ONE-HOT ENCODE CATEGORICAL FEATURES
    # --------------------------------------------------

    sandbox_encoded = pd.get_dummies(
        sandbox_model_features,
        columns=[
            "country",
            "channel",
            "entry_mode"
        ]
    )


    # --------------------------------------------------
    # 3. ALIGN TO EXACT TRAINED MODEL COLUMNS
    # --------------------------------------------------

    sandbox_encoded = (
        sandbox_encoded.reindex(
            columns=
                X_test_encoded.columns,
            fill_value=0
        )
    )


    # --------------------------------------------------
    # 4. MODEL RISK SCORE
    # --------------------------------------------------

    risk_score = float(
        model.get_booster().predict(
            xgb.DMatrix(
                sandbox_encoded,
                feature_names=sandbox_encoded.columns.tolist()
            )
        )[0]
    )


    # --------------------------------------------------
    # 5. THREE-WAY DECISION POLICY
    # --------------------------------------------------

    if risk_score >= Decline_Threshold:

        decision = "DECLINE"

    elif risk_score >= Review_Threshold:

        decision = "REVIEW"

    else:

        decision = "APPROVE"


    # --------------------------------------------------
    # 6. LOCAL SHAP VALUES
    # --------------------------------------------------

    shap_values = (
        sandbox_explainer.shap_values(
            sandbox_encoded
        )
    )


    local_shap = shap_values[0]


    # --------------------------------------------------
    # 7. RANK ALL SHAP FEATURES
    # --------------------------------------------------

    shap_ranked = []


    for feature_name, shap_value in zip(
        sandbox_encoded.columns,
        local_shap
    ):

        shap_ranked.append(
            {
                "feature":
                    feature_name,

                "value":
                    sandbox_encoded[
                        feature_name
                    ].iloc[0],

                "shap_value":
                    float(shap_value)
            }
        )


    # --------------------------------------------------
    # 8. KEEP ONLY BUSINESS-SAFE FEATURES
    # --------------------------------------------------

    safe_shap = []


    for item in shap_ranked:

        feature_name = item["feature"]


        if feature_name in safe_feature_map:

            safe_shap.append(
                {
                    "technical_feature":
                        feature_name,

                    "business_feature":
                        safe_feature_map[
                            feature_name
                        ],

                    "shap_value":
                        item[
                            "shap_value"
                        ]
                }
            )


    # --------------------------------------------------
    # 9. BUSINESS-FACING FEATURE VALUES
    # --------------------------------------------------

    feature_row = (
        sandbox_model_features.iloc[0]
    )

    txn_datetime = pd.Timestamp(
        transaction_datetime
    )


    def get_business_value(
        technical_feature
    ):

        if technical_feature == "CNP_flag":

            return (
                "Yes"
                if feature_row[
                    "CNP_flag"
                ] == 1
                else "No"
            )


        if technical_feature == "previous_device_uses":

            return int(
                feature_row[
                    "previous_device_uses"
                ]
            )


        if technical_feature == "transaction_hour":

            return txn_datetime.strftime(
                "%I:%M %p"
            )


        if technical_feature == "previous_merchant_uses":

            return int(
                feature_row[
                    "previous_merchant_uses"
                ]
            )


        if technical_feature == "amount_vs_customer_avg_filled":

            return (
                f"{feature_row[
                    'amount_vs_customer_avg_filled'
                ]:.3f}X"
            )


        if technical_feature == "channel_CNP":

            return feature_row[
                "channel"
            ]


        if technical_feature == "velocity_10min":

            return int(
                feature_row[
                    "velocity_10min"
                ]
            )


        if technical_feature == "new_device":

            return (
                "Yes"
                if feature_row[
                    "new_device"
                ] == 1
                else "No"
            )


        if technical_feature == "new_merchant":

            return (
                "Yes"
                if feature_row[
                    "new_merchant"
                ] == 1
                else "No"
            )


        if technical_feature == "time_since_last_transaction_filled":

            minutes = float(
                feature_row[
                    "time_since_last_transaction_filled"
                ]
            )


            if minutes >= 999999:

                return (
                    "No previous transaction history"
                )


            seconds = minutes * 60


            if minutes < 1:

                return (
                    f"{seconds:.3f} seconds"
                )

            elif minutes < 60:

                return (
                    f"{minutes:.3f} minutes"
                )

            elif minutes < 1440:

                return (
                    f"{minutes / 60:.3f} hours"
                )

            else:

                return (
                    f"{minutes / 1440:.3f} days"
                )


        return None


    # --------------------------------------------------
    # 10. ATTACH BUSINESS VALUES
    # --------------------------------------------------

    for item in safe_shap:

        item["business_value"] = (
            get_business_value(
                item[
                    "technical_feature"
                ]
            )
        )


    # --------------------------------------------------
    # 11. RISK-INCREASING INDICATORS
    # --------------------------------------------------

    risk_indicators = sorted(
        [
            item
            for item in safe_shap
            if item[
                "shap_value"
            ] > 0
        ],
        key=lambda x:
            x["shap_value"],
        reverse=True
    )[:5]


    # --------------------------------------------------
    # 12. RISK-REDUCING INDICATORS
    # --------------------------------------------------

    risk_reducing_indicators = sorted(
        [
            item
            for item in safe_shap
            if item[
                "shap_value"
            ] < 0
        ],
        key=lambda x:
            x["shap_value"]
    )[:3]


    # --------------------------------------------------
    # 13. JSON-READY OUTPUTS
    # --------------------------------------------------

    risk_indicators_json = [

        {
            "indicator":
                item[
                    "business_feature"
                ],

            "value":
                item[
                    "business_value"
                ],

            "shap_contribution":
                round(
                    item[
                        "shap_value"
                    ],
                    2
                )
        }

        for item in risk_indicators
    ]


    risk_reducing_indicators_json = [

        {
            "indicator":
                item[
                    "business_feature"
                ],

            "value":
                item[
                    "business_value"
                ],

            "shap_contribution":
                round(
                    item[
                        "shap_value"
                    ],
                    2
                )
        }

        for item in risk_reducing_indicators
    ]


    # --------------------------------------------------
    # 14. RETURN COMPLETE FRAUD RESULT
    # --------------------------------------------------

    return {

        "features":
            sandbox_model_features,

        "encoded_features":
            sandbox_encoded,

        "risk_score":
            risk_score,

        "decision":
            decision,

        "shap_values":
            shap_values,

        "safe_shap":
            safe_shap,

        "risk_indicators":
            risk_indicators,

        "risk_reducing_indicators":
            risk_reducing_indicators,

        "risk_indicators_json":
            risk_indicators_json,

        "risk_reducing_indicators_json":
            risk_reducing_indicators_json
    }



def build_sandbox_evidence(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    sandbox_result = (
        run_fraud_sandbox(
            customer_id=customer_id,
            amount=amount,
            merchant_id=merchant_id,
            country=country,
            channel=channel,
            entry_mode=entry_mode,
            device_id=device_id,
            transaction_datetime=
                transaction_datetime
        )
    )


    txn_datetime = pd.Timestamp(
        transaction_datetime
    )


    evidence = {

        "transaction": {

            "customer_id":
                int(customer_id),

            "merchant_id":
                int(merchant_id),

            "amount":
                f"₹{float(amount):,.3f}",

            "transaction_time":
                txn_datetime.strftime(
                    "%I:%M %p"
                ),

            "country":
                country,

            "channel":
                channel,

            "entry_mode":
                entry_mode,

            "decision":
                sandbox_result[
                    "decision"
                ],

            "model_risk_score":
                round(
                    sandbox_result[
                        "risk_score"
                    ],
                    6
                )
        },


        "risk_indicators":
            sandbox_result[
                "risk_indicators_json"
            ],


        "risk_reducing_indicators":
            sandbox_result[
                "risk_reducing_indicators_json"
            ]
    }


    return evidence



def build_investigator_prompt(
    evidence,
    permitted_actions
):

    evidence_json = json.dumps(
        evidence,
        indent=2
    )

    permitted_actions_json = json.dumps(
        permitted_actions,
        indent=2
    )


    prompt = f"""
You are an AI assistant supporting a payment fraud investigator.

Your job is to summarize the supplied transaction evidence and provide
an operational recommendation based ONLY on the evidence and permitted
actions supplied below.

IMPORTANT RULES:

1. Do NOT state that the transaction is definitely fraudulent or legitimate.

2. Do NOT invent:
   - customer intent
   - travel history
   - merchant facts
   - device ownership
   - account compromise
   - customer behavior
   - any information not explicitly supplied.

3. Clearly distinguish:
   - risk-increasing indicators
   - risk-reducing indicators

4. Do NOT contradict the decision engine.

5. Recommend ONLY actions present in the permitted action list.

6. If both risk-increasing and risk-reducing indicators are present,
   explicitly state that the transaction contains mixed signals.

7. Keep the response concise, factual, and operational.

8. The model risk score is NOT a calibrated probability of fraud.
   Do not describe it as a percentage probability that the transaction
   is fraudulent.


TRANSACTION EVIDENCE:

{evidence_json}


PERMITTED ACTIONS:

{permitted_actions_json}


Return the response using EXACTLY these sections:

Investigation Summary

Key Risk Indicators

Risk-Reducing Indicators

Recommended Action
"""


    return prompt



def build_deterministic_fallback(
    evidence,
    permitted_actions
):

    transaction = evidence[
        "transaction"
    ]

    risk_indicators = evidence[
        "risk_indicators"
    ]

    reducing_indicators = evidence[
        "risk_reducing_indicators"
    ]


    # --------------------------------------------------
    # INVESTIGATION SUMMARY
    # --------------------------------------------------

    if (
        len(risk_indicators) > 0
        and
        len(reducing_indicators) > 0
    ):

        signal_summary = (
            "The transaction contains mixed "
            "risk-increasing and risk-reducing "
            "signals."
        )

    elif len(risk_indicators) > 0:

        signal_summary = (
            "The transaction contains "
            "risk-increasing indicators."
        )

    elif len(reducing_indicators) > 0:

        signal_summary = (
            "The transaction contains "
            "risk-reducing indicators."
        )

    else:

        signal_summary = (
            "No material business-facing "
            "risk indicators were identified."
        )


    investigation_summary = (
        f"The decision engine returned "
        f"{transaction['decision']} with a "
        f"model risk score of "
        f"{transaction['model_risk_score']}. "
        f"{signal_summary} "
        f"The model score should not be "
        f"interpreted as a calibrated fraud "
        f"probability."
    )


    # --------------------------------------------------
    # FORMAT INDICATORS
    # --------------------------------------------------

    if risk_indicators:

        risk_text = "\n".join(
            [
                (
                    f"- {item['indicator']}: "
                    f"{item['value']}"
                )
                for item
                in risk_indicators
            ]
        )

    else:

        risk_text = (
            "- No business-facing "
            "risk-increasing indicators."
        )


    if reducing_indicators:

        reducing_text = "\n".join(
            [
                (
                    f"- {item['indicator']}: "
                    f"{item['value']}"
                )
                for item
                in reducing_indicators
            ]
        )

    else:

        reducing_text = (
            "- No business-facing "
            "risk-reducing indicators."
        )


    # --------------------------------------------------
    # PERMITTED ACTION
    # --------------------------------------------------

    if permitted_actions:

        action_text = "\n".join(
            [
                f"- {action}"
                for action
                in permitted_actions
            ]
        )

    else:

        action_text = (
            "- No permitted action supplied."
        )


    # --------------------------------------------------
    # FINAL FALLBACK RESPONSE
    # --------------------------------------------------

    fallback_text = f"""
Investigation Summary

{investigation_summary}

Key Risk Indicators

{risk_text}

Risk-Reducing Indicators

{reducing_text}

Recommended Action

{action_text}
""".strip()


    return fallback_text



def call_gemini_with_retry(
    prompt,
    evidence,
    permitted_actions,
    max_retries=3,
    wait_seconds=3
):

    for attempt in range(
        max_retries
    ):

        try:

            response = (
                client.models.generate_content(
                    model="gemini-3.7-flash",
                    contents=prompt
                )
            )


            return {

                "status":
                    "success",

                "source":
                    "Gemini",

                "text":
                    response.text
            }


        except Exception as e:

            print(
                f"Gemini attempt "
                f"{attempt + 1}/{max_retries} "
                f"failed:"
            )

            print(e)


            if (
                attempt
                <
                max_retries - 1
            ):

                time.sleep(
                    wait_seconds
                )


    # --------------------------------------------------
    # FALLBACK AFTER ALL RETRIES FAIL
    # --------------------------------------------------

    fallback_text = (
        build_deterministic_fallback(
            evidence=evidence,
            permitted_actions=
                permitted_actions
        )
    )


    return {

        "status":
            "fallback",

        "source":
            "Deterministic Fraud Engine",

        "text":
            fallback_text
    }



def standardize_copilot_output(
    genai_result,
    evidence
):

    return {

        "status":
            genai_result[
                "status"
            ],

        "source":
            genai_result[
                "source"
            ],

        "decision":
            evidence[
                "transaction"
            ][
                "decision"
            ],

        "model_risk_score":
            evidence[
                "transaction"
            ][
                "model_risk_score"
            ],

        "risk_indicators":
            evidence[
                "risk_indicators"
            ],

        "risk_reducing_indicators":
            evidence[
                "risk_reducing_indicators"
            ],

        "investigator_explanation":
            genai_result[
                "text"
            ]
    }



def validate_sandbox_inputs(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    valid_entry_modes = {

        "CNP":
            [
                "ECOM",
                "MANUAL"
            ],

        "POS":
            [
                "CHIP",
                "SWIPE"
            ],

        "CONTACTLESS":
            [
                "TAP"
            ]
    }


    # --------------------------------------------------
    # CUSTOMER
    # --------------------------------------------------

    if int(customer_id) not in set(
        df["customer_id"].unique()
    ):

        raise ValueError(
            f"Customer ID {customer_id} "
            "does not exist."
        )


    # --------------------------------------------------
    # AMOUNT
    # --------------------------------------------------

    if float(amount) <= 0:

        raise ValueError(
            "Transaction amount must be "
            "greater than zero."
        )


    # --------------------------------------------------
    # MERCHANT
    # --------------------------------------------------

    if int(merchant_id) not in set(
        df["merchant_id"].unique()
    ):

        raise ValueError(
            f"Merchant ID {merchant_id} "
            "does not exist."
        )


    # --------------------------------------------------
    # COUNTRY
    # --------------------------------------------------

    if country not in set(
        df["country"].unique()
    ):

        raise ValueError(
            f"Country {country} "
            "does not exist."
        )


    # --------------------------------------------------
    # CHANNEL
    # --------------------------------------------------

    if channel not in valid_entry_modes:

        raise ValueError(
            f"Invalid channel: {channel}"
        )


    # --------------------------------------------------
    # ENTRY MODE
    # --------------------------------------------------

    if entry_mode not in (
        valid_entry_modes[
            channel
        ]
    ):

        raise ValueError(
            f"Entry mode {entry_mode} "
            f"is not valid for "
            f"channel {channel}."
        )


    # --------------------------------------------------
    # DEVICE
    # --------------------------------------------------

    if device_id is None:

        raise ValueError(
            "Device ID cannot be None."
        )


    # --------------------------------------------------
    # DATETIME
    # --------------------------------------------------

    pd.Timestamp(
        transaction_datetime
    )


    return True



def analyze_transaction_end_to_end(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    # --------------------------------------------------
    # 1. VALIDATE INPUTS
    # --------------------------------------------------

    validate_sandbox_inputs(
        customer_id=customer_id,
        amount=amount,
        merchant_id=merchant_id,
        country=country,
        channel=channel,
        entry_mode=entry_mode,
        device_id=device_id,
        transaction_datetime=
            transaction_datetime
    )


    # --------------------------------------------------
    # 2. RUN FRAUD MODEL + SHAP
    # --------------------------------------------------

    sandbox_result = (
        run_fraud_sandbox(
            customer_id=customer_id,
            amount=amount,
            merchant_id=merchant_id,
            country=country,
            channel=channel,
            entry_mode=entry_mode,
            device_id=device_id,
            transaction_datetime=
                transaction_datetime
        )
    )


    # --------------------------------------------------
    # 3. BUILD INVESTIGATOR EVIDENCE
    # --------------------------------------------------

    evidence = (
        build_sandbox_evidence(
            customer_id=customer_id,
            amount=amount,
            merchant_id=merchant_id,
            country=country,
            channel=channel,
            entry_mode=entry_mode,
            device_id=device_id,
            transaction_datetime=
                transaction_datetime
        )
    )


    # --------------------------------------------------
    # 4. GET PERMITTED ACTIONS
    # --------------------------------------------------

    decision = sandbox_result[
        "decision"
    ]

    permitted_actions = (
        action_policy[
            decision
        ]
    )


    # --------------------------------------------------
    # 5. BUILD INVESTIGATOR PROMPT
    # --------------------------------------------------

    prompt = (
        build_investigator_prompt(
            evidence=evidence,
            permitted_actions=
                permitted_actions
        )
    )


    # --------------------------------------------------
    # 6. GEMINI OR DETERMINISTIC FALLBACK
    # --------------------------------------------------

    genai_result = (
        call_gemini_with_retry(
            prompt=prompt,
            evidence=evidence,
            permitted_actions=
                permitted_actions
        )
    )


    # --------------------------------------------------
    # 7. STANDARDIZE COPILOT OUTPUT
    # --------------------------------------------------

    standardized_copilot = (
        standardize_copilot_output(
            genai_result=
                genai_result,
            evidence=
                evidence
        )
    )


    # --------------------------------------------------
    # 8. FINAL STRUCTURED RESULT
    # --------------------------------------------------

    return {

        "risk_decision": {

            "model_risk_score":
                round(
                    sandbox_result[
                        "risk_score"
                    ],
                    6
                ),

            "decision":
                sandbox_result[
                    "decision"
                ]
        },


        "explainability": {

            "risk_indicators":
                sandbox_result[
                    "risk_indicators_json"
                ],

            "risk_reducing_indicators":
                sandbox_result[
                    "risk_reducing_indicators_json"
                ]
        },


        "investigator_copilot":
            standardized_copilot
    }



def get_merchant_accepted_currencies(
    merchant_id
):

    merchant_id = int(
        merchant_id
    )


    if (
        merchant_id
        not in
        merchant_accepted_currencies
    ):

        raise ValueError(
            f"Merchant ID {merchant_id} "
            "does not exist."
        )


    return (
        merchant_accepted_currencies[
            merchant_id
        ]
    )



def compare_merchant_currency_options(
    merchant_id,
    normalized_amount_inr
):

    merchant_id = int(
        merchant_id
    )

    normalized_amount_inr = float(
        normalized_amount_inr
    )


    # --------------------------------------------------
    # VALIDATE AMOUNT
    # --------------------------------------------------

    if normalized_amount_inr <= 0:

        raise ValueError(
            "Transaction value must be "
            "greater than zero."
        )


    # --------------------------------------------------
    # GET MERCHANT ACCEPTED CURRENCIES
    # --------------------------------------------------

    accepted_currencies = (
        get_merchant_accepted_currencies(
            merchant_id
        )
    )


    # --------------------------------------------------
    # BUILD PURE FX OPTIONS
    # --------------------------------------------------

    currency_options = []


    for currency in accepted_currencies:

        fx_rate = float(
            fx_to_inr[
                currency
            ]
        )


        currency_amount = (
            normalized_amount_inr
            /
            fx_rate
        )


        currency_options.append(
            {
                "merchant_id":
                    merchant_id,

                "currency":
                    currency,

                "fx_rate_to_inr":
                    fx_rate,

                "currency_amount":
                    currency_amount,

                "normalized_amount_inr":
                    normalized_amount_inr
            }
        )


    return pd.DataFrame(
        currency_options
    )



def compare_customer_currency_options(
    customer_id,
    merchant_id,
    normalized_amount_inr
):

    customer_id = int(customer_id)
    merchant_id = int(merchant_id)

    normalized_amount_inr = float(
        normalized_amount_inr
    )


    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if normalized_amount_inr <= 0:

        raise ValueError(
            "Transaction value must be "
            "greater than zero."
        )


    if customer_id not in customer_home_country:

        raise ValueError(
            f"Customer ID {customer_id} "
            "does not exist."
        )


    if merchant_id not in merchant_primary_country:

        raise ValueError(
            f"Merchant ID {merchant_id} "
            "does not exist."
        )


    # --------------------------------------------------
    # CUSTOMER HOME CURRENCY
    # --------------------------------------------------

    customer_country = (
        customer_home_country[
            customer_id
        ]
    )

    customer_currency = (
        country_default_currency[
            customer_country
        ]
    )


    # --------------------------------------------------
    # MERCHANT HOME CURRENCY
    # --------------------------------------------------

    merchant_country = (
        merchant_primary_country[
            merchant_id
        ]
    )

    merchant_currency = (
        country_default_currency[
            merchant_country
        ]
    )


    # --------------------------------------------------
    # PURE FX OPTIONS
    # --------------------------------------------------

    comparison = (
        compare_merchant_currency_options(
            merchant_id=
                merchant_id,

            normalized_amount_inr=
                normalized_amount_inr
        )
    )


    # --------------------------------------------------
    # ADD HOME-CURRENCY CONTEXT
    # --------------------------------------------------

    comparison[
        "customer_home_currency"
    ] = customer_currency

    comparison[
        "merchant_home_currency"
    ] = merchant_currency


    # --------------------------------------------------
    # CUSTOMER-SIDE FX MARKUP
    # --------------------------------------------------

    comparison[
        "customer_fx_markup_rate"
    ] = comparison[
        "currency"
    ].apply(
        lambda selected_currency:
            fx_pair_markup[
                (
                    customer_currency,
                    selected_currency
                )
            ]
    )


    # --------------------------------------------------
    # MERCHANT-SIDE PRICING MARKUP
    # --------------------------------------------------

    comparison[
        "merchant_pricing_markup_rate"
    ] = comparison[
        "currency"
    ].apply(
        lambda selected_currency:
            merchant_currency_markup[
                merchant_id
            ][
                selected_currency
            ]
    )


    # --------------------------------------------------
    # TOTAL MARKUP
    # --------------------------------------------------
    #
    # Current synthetic project assumption:
    # simple additive approximation.
    # --------------------------------------------------

    comparison[
        "total_markup_rate"
    ] = (
        comparison[
            "customer_fx_markup_rate"
        ]
        +
        comparison[
            "merchant_pricing_markup_rate"
        ]
    )


    # --------------------------------------------------
    # ESTIMATED MARKUP COST
    # --------------------------------------------------

    comparison[
        "estimated_markup_cost_inr"
    ] = (
        normalized_amount_inr
        *
        comparison[
            "total_markup_rate"
        ]
    )


    # --------------------------------------------------
    # ESTIMATED TOTAL ECONOMIC COST
    # --------------------------------------------------

    comparison[
        "estimated_total_cost_inr"
    ] = (
        normalized_amount_inr
        +
        comparison[
            "estimated_markup_cost_inr"
        ]
    )


    # --------------------------------------------------
    # RANK CURRENCY OPTIONS
    # --------------------------------------------------

    comparison = (
        comparison
        .sort_values(
            "estimated_total_cost_inr",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )


    comparison[
        "cost_rank"
    ] = (
        np.arange(
            1,
            len(comparison) + 1
        )
    )


    return comparison



def recommend_payment_currency(
    customer_id,
    merchant_id,
    normalized_amount_inr
):

    comparison = (
        compare_customer_currency_options(
            customer_id=
                customer_id,

            merchant_id=
                merchant_id,

            normalized_amount_inr=
                normalized_amount_inr
        )
    )


    # --------------------------------------------------
    # BEST OPTION
    # --------------------------------------------------

    best_option = (
        comparison.iloc[0]
    )


    # --------------------------------------------------
    # SECOND-BEST OPTION
    # --------------------------------------------------

    if len(comparison) > 1:

        second_best_option = (
            comparison.iloc[1]
        )

        savings_vs_next_best_inr = (
            second_best_option[
                "estimated_total_cost_inr"
            ]
            -
            best_option[
                "estimated_total_cost_inr"
            ]
        )

    else:

        savings_vs_next_best_inr = 0.0


    # --------------------------------------------------
    # RETURN RECOMMENDATION
    # --------------------------------------------------

    return {

        "customer_id":
            int(customer_id),

        "merchant_id":
            int(merchant_id),

        "customer_home_currency":
            best_option[
                "customer_home_currency"
            ],

        "merchant_home_currency":
            best_option[
                "merchant_home_currency"
            ],

        "recommended_currency":
            best_option[
                "currency"
            ],

        "estimated_total_cost_inr":
            float(
                best_option[
                    "estimated_total_cost_inr"
                ]
            ),

        "total_markup_rate":
            float(
                best_option[
                    "total_markup_rate"
                ]
            ),

        "savings_vs_next_best_inr":
            float(
                savings_vs_next_best_inr
            ),

        "accepted_currencies":
            comparison[
                "currency"
            ].tolist()
    }



def build_currency_recommendation_evidence(
    customer_id,
    merchant_id,
    normalized_amount_inr
):

    comparison = (
        compare_customer_currency_options(
            customer_id=customer_id,
            merchant_id=merchant_id,
            normalized_amount_inr=normalized_amount_inr
        )
    )


    recommendation = (
        recommend_payment_currency(
            customer_id=customer_id,
            merchant_id=merchant_id,
            normalized_amount_inr=normalized_amount_inr
        )
    )


    currency_options = []


    for _, row in comparison.iterrows():

        currency_options.append(
            {
                "currency":
                    row["currency"],

                "currency_amount":
                    round(
                        float(
                            row["currency_amount"]
                        ),
                        4
                    ),

                "customer_fx_markup_percent":
                    round(
                        float(
                            row[
                                "customer_fx_markup_rate"
                            ]
                        ) * 100,
                        3
                    ),

                "merchant_pricing_markup_percent":
                    round(
                        float(
                            row[
                                "merchant_pricing_markup_rate"
                            ]
                        ) * 100,
                        3
                    ),

                "total_markup_percent":
                    round(
                        float(
                            row[
                                "total_markup_rate"
                            ]
                        ) * 100,
                        3
                    ),

                "estimated_total_cost_inr":
                    round(
                        float(
                            row[
                                "estimated_total_cost_inr"
                            ]
                        ),
                        2
                    ),

                "cost_rank":
                    int(
                        row[
                            "cost_rank"
                        ]
                    )
            }
        )


    return {

        "customer_id":
            int(customer_id),

        "merchant_id":
            int(merchant_id),

        "base_transaction_value_inr":
            round(
                float(
                    normalized_amount_inr
                ),
                2
            ),

        "customer_home_currency":
            recommendation[
                "customer_home_currency"
            ],

        "merchant_home_currency":
            recommendation[
                "merchant_home_currency"
            ],

        "accepted_currencies":
            recommendation[
                "accepted_currencies"
            ],

        "recommended_currency":
            recommendation[
                "recommended_currency"
            ],

        "estimated_total_cost_inr":
            round(
                recommendation[
                    "estimated_total_cost_inr"
                ],
                2
            ),

        "savings_vs_next_best_inr":
            round(
                recommendation[
                    "savings_vs_next_best_inr"
                ],
                2
            ),

        "currency_options":
            currency_options
    }



def build_currency_recommendation_prompt(
    evidence
):

    evidence_json = json.dumps(
        evidence,
        indent=2
    )


    prompt = f"""
You are a Payment Currency Recommendation Assistant.

Your job is to explain a currency recommendation that has
already been calculated by a deterministic payment engine.

The deterministic engine has already evaluated:

- merchant accepted currencies
- customer home currency
- merchant home currency
- synthetic FX conversion assumptions
- customer-side FX markup
- merchant-side pricing markup
- estimated total economic cost
- currency cost ranking

You MUST use only the supplied evidence.

STRICT RULES:

1. Do NOT recalculate exchange rates, markups, costs, rankings,
   or savings.

2. Do NOT change the recommended currency.

3. Do NOT assume that the currency with the smallest numerical
   transaction amount is the cheapest option.

4. Do NOT invent issuer fees, card-network fees, merchant fees,
   taxes, spreads, exchange rates, or customer preferences.

5. Clearly distinguish:
   - customer-side FX markup
   - merchant-side pricing markup
   - total estimated cost

6. Explain why the recommended currency ranks first using only
   the supplied evidence.

7. If another currency has zero merchant markup but still ranks
   worse, explain that other cost components can make its total
   estimated cost higher.

8. Treat all FX rates and markup assumptions as synthetic project
   assumptions. Do not present them as actual issuer, merchant,
   Mastercard, Visa, or market pricing.

9. Keep the explanation concise, factual, and customer-friendly.

10. The deterministic recommendation engine is authoritative.
    Your role is explanation, not decision-making.

SUPPLIED EVIDENCE:

{evidence_json}

Return EXACTLY these sections:

Recommended Currency:
Why This Currency:
Cost Comparison:
Important Note:
"""

    return prompt



def call_currency_gemini_with_retry(
    prompt,
    evidence,
    max_retries=3,
    wait_seconds=3
):

    # --------------------------------------------------
    # TRY GEMINI
    # --------------------------------------------------

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            response = (
                client.models.generate_content(
                    model="gemini-3.7-flash",
                    contents=prompt
                )
            )


            return {

                "status":
                    "success",

                "source":
                    "Gemini",

                "text":
                    response.text
            }


        except Exception as error:

            print(
                f"Currency Gemini attempt "
                f"{attempt} failed:",
                error
            )


            if attempt < max_retries:

                time.sleep(
                    wait_seconds
                )


    # --------------------------------------------------
    # DETERMINISTIC FALLBACK
    # --------------------------------------------------

    recommended_currency = (
        evidence[
            "recommended_currency"
        ]
    )

    customer_currency = (
        evidence[
            "customer_home_currency"
        ]
    )

    merchant_currency = (
        evidence[
            "merchant_home_currency"
        ]
    )

    estimated_cost = (
        evidence[
            "estimated_total_cost_inr"
        ]
    )

    savings = (
        evidence[
            "savings_vs_next_best_inr"
        ]
    )


    fallback_text = f"""
Recommended Currency:
{recommended_currency}

Why This Currency:
The deterministic currency engine ranked {recommended_currency}
as the lowest estimated-cost option among the currencies evaluated
for this transaction.

Cost Comparison:
The estimated total economic cost is ₹{estimated_cost:,.2f}.
The estimated saving versus the next-best currency option is
₹{savings:,.2f}.

Important Note:
The customer home currency is {customer_currency} and the merchant
home currency is {merchant_currency}. FX rates and markup assumptions
used here are synthetic project assumptions and are not actual issuer,
merchant, Mastercard, Visa, or market pricing.
""".strip()


    return {

        "status":
            "fallback",

        "source":
            "Deterministic Currency Engine",

        "text":
            fallback_text
    }



def standardize_currency_assistant_output(
    genai_result,
    evidence
):

    return {

        # --------------------------------------------------
        # ASSISTANT EXECUTION STATUS
        # --------------------------------------------------

        "status":
            genai_result[
                "status"
            ],

        "source":
            genai_result[
                "source"
            ],


        # --------------------------------------------------
        # DETERMINISTIC RECOMMENDATION
        # --------------------------------------------------

        "recommended_currency":
            evidence[
                "recommended_currency"
            ],

        "customer_home_currency":
            evidence[
                "customer_home_currency"
            ],

        "merchant_home_currency":
            evidence[
                "merchant_home_currency"
            ],


        # --------------------------------------------------
        # ECONOMIC RESULT
        # --------------------------------------------------

        "estimated_total_cost_inr":
            evidence[
                "estimated_total_cost_inr"
            ],

        "savings_vs_next_best_inr":
            evidence[
                "savings_vs_next_best_inr"
            ],


        # --------------------------------------------------
        # FULL DETERMINISTIC EVIDENCE
        # --------------------------------------------------

        "accepted_currencies":
            evidence[
                "accepted_currencies"
            ],

        "currency_options":
            evidence[
                "currency_options"
            ],


        # --------------------------------------------------
        # GENAI / FALLBACK EXPLANATION
        # --------------------------------------------------

        "assistant_explanation":
            genai_result[
                "text"
            ]
    }



def add_customer_currency_display(
    standardized_output
):

    # Work on a deep copy so that nested
    # currency_options are not modified
    # elsewhere accidentally.

    result = copy.deepcopy(
        standardized_output
    )


    # --------------------------------------------------
    # CUSTOMER HOME CURRENCY
    # --------------------------------------------------

    customer_currency = (
        result[
            "customer_home_currency"
        ]
    )


    # --------------------------------------------------
    # FX RATE USED ONLY FOR PRESENTATION
    # --------------------------------------------------

    customer_fx_rate_to_inr = float(
        fx_to_inr[
            customer_currency
        ]
    )


    # --------------------------------------------------
    # CONVERT RECOMMENDED TOTAL COST
    # --------------------------------------------------

    result[
        "estimated_total_cost_customer_currency"
    ] = round(

        result[
            "estimated_total_cost_inr"
        ]
        /
        customer_fx_rate_to_inr,

        2
    )


    # --------------------------------------------------
    # CONVERT SAVINGS
    # --------------------------------------------------

    result[
        "savings_vs_next_best_customer_currency"
    ] = round(

        result[
            "savings_vs_next_best_inr"
        ]
        /
        customer_fx_rate_to_inr,

        2
    )


    # --------------------------------------------------
    # ADD DISPLAY CURRENCY CODE
    # --------------------------------------------------

    result[
        "display_currency"
    ] = customer_currency


    # --------------------------------------------------
    # CONVERT EACH OPTION'S TOTAL COST
    # --------------------------------------------------

    for option in result[
        "currency_options"
    ]:

        option[
            "estimated_total_cost_customer_currency"
        ] = round(

            option[
                "estimated_total_cost_inr"
            ]
            /
            customer_fx_rate_to_inr,

            2
        )


    return result



def analyze_currency_recommendation_end_to_end(
    customer_id,
    merchant_id,
    normalized_amount_inr
):

    customer_id = int(
        customer_id
    )

    merchant_id = int(
        merchant_id
    )

    normalized_amount_inr = float(
        normalized_amount_inr
    )


    # --------------------------------------------------
    # 1. VALIDATE INPUTS
    # --------------------------------------------------

    if customer_id not in customer_home_country:

        raise ValueError(
            f"Customer ID {customer_id} "
            "does not exist."
        )


    if merchant_id not in merchant_primary_country:

        raise ValueError(
            f"Merchant ID {merchant_id} "
            "does not exist."
        )


    if normalized_amount_inr <= 0:

        raise ValueError(
            "Transaction value must be "
            "greater than zero."
        )


    # --------------------------------------------------
    # 2. BUILD DETERMINISTIC EVIDENCE
    # --------------------------------------------------

    evidence = (
        build_currency_recommendation_evidence(
            customer_id=
                customer_id,

            merchant_id=
                merchant_id,

            normalized_amount_inr=
                normalized_amount_inr
        )
    )


    # --------------------------------------------------
    # 3. BUILD GENAI PROMPT
    # --------------------------------------------------

    prompt = (
        build_currency_recommendation_prompt(
            evidence
        )
    )


    # --------------------------------------------------
    # 4. GEMINI OR DETERMINISTIC FALLBACK
    # --------------------------------------------------

    genai_result = (
        call_currency_gemini_with_retry(
            prompt=
                prompt,

            evidence=
                evidence
        )
    )


    # --------------------------------------------------
    # 5. STANDARDIZE OUTPUT
    # --------------------------------------------------

    standardized_output = (
        standardize_currency_assistant_output(
            genai_result=
                genai_result,

            evidence=
                evidence
        )
    )


    # --------------------------------------------------
    # 6. ADD CUSTOMER-CURRENCY DISPLAY
    # --------------------------------------------------

    final_output = (
        add_customer_currency_display(
            standardized_output
        )
    )


    return final_output



def check_merchant_restriction(
    merchant_id
):

    merchant_id = int(
        merchant_id
    )


    # --------------------------------------------------
    # 1. RESTRICTED MERCHANT
    # --------------------------------------------------

    if merchant_id in restricted_merchant_registry:

        merchant_record = (
            restricted_merchant_registry[
                merchant_id
            ]
        )


        return {

            "merchant_id":
                merchant_id,

            "merchant_status":
                "RESTRICTED",

            "platform_decision":
                "DECLINE",

            "restriction_reason":
                str(
                    merchant_record[
                        "restriction_reason"
                    ]
                ),

            "merchant_country":
                str(
                    merchant_record[
                        "merchant_country"
                    ]
                ),

            "merchant_home_currency":
                str(
                    merchant_record[
                        "merchant_home_currency"
                    ]
                ),

            "proceed_to_fraud_engine":
                False,

            "proceed_to_currency_engine":
                False
        }


    # --------------------------------------------------
    # 2. ACTIVE MERCHANT
    # --------------------------------------------------

    if merchant_id in merchant_primary_country:

        merchant_country = str(
            merchant_primary_country[
                merchant_id
            ]
        )

        merchant_home_currency = str(
            country_default_currency[
                merchant_country
            ]
        )


        return {

            "merchant_id":
                merchant_id,

            "merchant_status":
                "ACTIVE",

            "platform_decision":
                "PROCEED",

            "restriction_reason":
                None,

            "merchant_country":
                merchant_country,

            "merchant_home_currency":
                merchant_home_currency,

            "proceed_to_fraud_engine":
                True,

            "proceed_to_currency_engine":
                True
        }


    # --------------------------------------------------
    # 3. UNKNOWN MERCHANT
    # --------------------------------------------------

    return {

        "merchant_id":
            merchant_id,

        "merchant_status":
            "UNKNOWN",

        "platform_decision":
            "REVIEW",

        "restriction_reason":
            "MERCHANT_NOT_FOUND",

        "merchant_country":
            None,

        "merchant_home_currency":
            None,

        "proceed_to_fraud_engine":
            False,

        "proceed_to_currency_engine":
            False
    }



def analyze_payment_intelligence(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):

    # --------------------------------------------------
    # 1. MERCHANT CONTROL GATE
    # --------------------------------------------------

    merchant_control = (
        check_merchant_restriction(
            merchant_id
        )
    )


    # --------------------------------------------------
    # 2. STOP IF MERCHANT CANNOT PROCEED
    # --------------------------------------------------

    if (
        merchant_control[
            "platform_decision"
        ]
        !=
        "PROCEED"
    ):

        return {

            "transaction": {

                "customer_id":
                    int(customer_id),

                "merchant_id":
                    int(merchant_id),

                "amount":
                    float(amount),

                "country":
                    country,

                "channel":
                    channel,

                "entry_mode":
                    entry_mode,

                "device_id":
                    device_id,

                "transaction_datetime":
                    str(
                        pd.Timestamp(
                            transaction_datetime
                        )
                    )
            },


            "merchant_control":
                merchant_control,


            "platform_decision":
                merchant_control[
                    "platform_decision"
                ],


            "decision_source":
                "MERCHANT_RESTRICTION_CONTROL",


            "fraud_intelligence":
                None,


            "currency_intelligence":
                None
        }


    # --------------------------------------------------
    # 3. FRAUD INTELLIGENCE
    # --------------------------------------------------

    fraud_intelligence = (
        analyze_transaction_end_to_end(
            customer_id=
                customer_id,

            amount=
                amount,

            merchant_id=
                merchant_id,

            country=
                country,

            channel=
                channel,

            entry_mode=
                entry_mode,

            device_id=
                device_id,

            transaction_datetime=
                transaction_datetime
        )
    )


    # --------------------------------------------------
    # 4. CURRENCY INTELLIGENCE
    # --------------------------------------------------

    currency_intelligence = (
        analyze_currency_recommendation_end_to_end(
            customer_id=
                customer_id,

            merchant_id=
                merchant_id,

            normalized_amount_inr=
                amount
        )
    )


    # --------------------------------------------------
    # 5. FINAL PLATFORM RESULT
    # --------------------------------------------------

    return {

        "transaction": {

            "customer_id":
                int(customer_id),

            "merchant_id":
                int(merchant_id),

            "amount":
                float(amount),

            "country":
                country,

            "channel":
                channel,

            "entry_mode":
                entry_mode,

            "device_id":
                device_id,

            "transaction_datetime":
                str(
                    pd.Timestamp(
                        transaction_datetime
                    )
                )
        },


        "merchant_control":
            merchant_control,


        "platform_decision":
            fraud_intelligence[
                "risk_decision"
            ][
                "decision"
            ],


        "decision_source":
            "FRAUD_DECISION_ENGINE",


        "fraud_intelligence":
            fraud_intelligence,


        "currency_intelligence":
            currency_intelligence
    }





def recovery_health_check():
    """
    Basic structural health check after state restoration.
    """

    return {

        "dataset_shape":
            tuple(df.shape),

        "encoded_features":
            int(
                X_test_encoded.shape[1]
            ),

        "review_threshold":
            float(
                Review_Threshold
            ),

        "decline_threshold":
            float(
                Decline_Threshold
            ),

        "active_merchants":
            len(
                merchant_primary_country
            ),

        "restricted_merchants":
            len(
                restricted_merchant_registry
            ),

        "gemini_initialized":
            client is not None,

        "shap_initialized":
            sandbox_explainer
            is not None
    }



# ============================================================
# POST-POLICY RECOVERY EXTENSION
# Added after STEP 543
# ============================================================


# ------------------------------------------------------------
# REQUIRED STATE EXTENSION
#
# restore_runtime_state() reads REQUIRED_STATE_KEYS at runtime,
# so extending this global list here is sufficient.
# ------------------------------------------------------------

if "transaction_review_policy" not in REQUIRED_STATE_KEYS:
    REQUIRED_STATE_KEYS = (
        list(REQUIRED_STATE_KEYS)
        +
        ["transaction_review_policy"]
    )


# ------------------------------------------------------------
# TRANSACTION REVIEW POLICY EVALUATOR
# ------------------------------------------------------------

def evaluate_transaction_review_policy(
    transaction_features
):

    triggered_rules = []


    # --------------------------------------------------------
    # CUSTOMER HISTORY
    #
    # Existing behavioral engine returns the number of
    # transactions before the current transaction.
    # --------------------------------------------------------

    previous_customer_transactions = int(
        transaction_features[
            "previous_customer_transactions"
        ]
    )


    has_history = (
        previous_customer_transactions
        >
        0
    )


    # --------------------------------------------------------
    # RULE 1:
    # Transaction amount >= 2X historical customer average
    # --------------------------------------------------------

    amount_ratio = float(
        transaction_features[
            "amount_vs_customer_avg_filled"
        ]
    )


    amount_threshold = float(
        transaction_review_policy[
            "amount_vs_average"
        ][
            "threshold_multiplier"
        ]
    )


    if (
        transaction_review_policy[
            "amount_vs_average"
        ][
            "enabled"
        ]
        and
        has_history
        and
        amount_ratio >= amount_threshold
    ):

        triggered_rules.append(
            {
                "rule":
                    "amount_vs_average",

                "reason":
                    transaction_review_policy[
                        "amount_vs_average"
                    ][
                        "reason"
                    ],

                "observed_value":
                    round(
                        amount_ratio,
                        3
                    ),

                "threshold":
                    amount_threshold,

                "action":
                    "REVIEW"
            }
        )


    # --------------------------------------------------------
    # RULE 2:
    # Merchant entirely new to this customer
    # --------------------------------------------------------

    new_merchant = int(
        transaction_features[
            "new_merchant"
        ]
    )


    if (
        transaction_review_policy[
            "new_merchant"
        ][
            "enabled"
        ]
        and
        new_merchant
        ==
        transaction_review_policy[
            "new_merchant"
        ][
            "trigger_value"
        ]
    ):

        triggered_rules.append(
            {
                "rule":
                    "new_merchant",

                "reason":
                    transaction_review_policy[
                        "new_merchant"
                    ][
                        "reason"
                    ],

                "observed_value":
                    new_merchant,

                "threshold":
                    1,

                "action":
                    "REVIEW"
            }
        )


    # --------------------------------------------------------
    # FINAL POLICY RESULT
    # --------------------------------------------------------

    review_required = (
        len(
            triggered_rules
        )
        >
        0
    )


    return {

        "review_required":
            review_required,

        "policy_action":
            (
                "REVIEW"
                if review_required
                else
                "NO_OVERRIDE"
            ),

        "previous_customer_transactions":
            previous_customer_transactions,

        "trigger_count":
            len(
                triggered_rules
            ),

        "triggered_rules":
            triggered_rules
    }



# ------------------------------------------------------------
# POLICY DECISION ORCHESTRATOR V2
# ------------------------------------------------------------

def apply_policy_decision(
    merchant_control,
    fraud_intelligence,
    transaction_review_result=None
):


    # ========================================================
    # PRIORITY 1:
    # MERCHANT RESTRICTION CONTROL
    # ========================================================

    merchant_decision = (
        merchant_control[
            "platform_decision"
        ]
    )

    merchant_status = (
        merchant_control[
            "merchant_status"
        ]
    )


    # --------------------------------------------------------
    # Restricted merchant
    # --------------------------------------------------------

    if merchant_decision == "DECLINE":

        return {

            "final_decision":
                "DECLINE",

            "decision_source":
                "MERCHANT_RESTRICTION_CONTROL",

            "decision_reason":
                merchant_control[
                    "restriction_reason"
                ],

            "fraud_engine_considered":
                False,

            "transaction_review_policy_considered":
                False
        }


    # --------------------------------------------------------
    # Unknown merchant
    # --------------------------------------------------------

    if merchant_decision == "REVIEW":

        return {

            "final_decision":
                "REVIEW",

            "decision_source":
                "MERCHANT_RESTRICTION_CONTROL",

            "decision_reason":
                merchant_control[
                    "restriction_reason"
                ],

            "fraud_engine_considered":
                False,

            "transaction_review_policy_considered":
                False
        }


    # ========================================================
    # PRIORITY 2:
    # ACTIVE MERCHANT → EXISTING FRAUD ENGINE
    # ========================================================

    if merchant_status == "ACTIVE":

        fraud_decision = (
            fraud_intelligence[
                "risk_decision"
            ][
                "decision"
            ]
        )

        risk_score = float(
            fraud_intelligence[
                "risk_decision"
            ][
                "model_risk_score"
            ]
        )


        # ----------------------------------------------------
        # XGBOOST DECLINE MUST NEVER BE DOWNGRADED
        # ----------------------------------------------------

        if fraud_decision == "DECLINE":

            return {

                "final_decision":
                    "DECLINE",

                "decision_source":
                    "FRAUD_DECISION_ENGINE",

                "decision_reason":
                    "EXISTING_FRAUD_MODEL_DECISION",

                "fraud_engine_considered":
                    True,

                "transaction_review_policy_considered":
                    True,

                "model_risk_score":
                    risk_score
            }


        # ====================================================
        # PRIORITY 3:
        # DETERMINISTIC TRANSACTION REVIEW POLICY
        #
        # Either:
        #   Amount >= 2X average
        #   OR
        #   Entirely new merchant
        #
        # escalates transaction to REVIEW.
        # ====================================================

        if (
            transaction_review_result
            is not None

            and

            transaction_review_result[
                "review_required"
            ]
            is True
        ):

            triggered_reasons = [

                rule[
                    "reason"
                ]

                for rule in (
                    transaction_review_result[
                        "triggered_rules"
                    ]
                )
            ]


            return {

                "final_decision":
                    "REVIEW",

                "decision_source":
                    "TRANSACTION_REVIEW_POLICY",

                "decision_reason":
                    triggered_reasons,

                "fraud_engine_considered":
                    True,

                "transaction_review_policy_considered":
                    True,

                "model_risk_score":
                    risk_score,

                "model_original_decision":
                    fraud_decision,

                "review_trigger_count":
                    transaction_review_result[
                        "trigger_count"
                    ]
            }


        # ====================================================
        # NO POLICY OVERRIDE
        #
        # Keep original fraud-engine decision.
        # ====================================================

        return {

            "final_decision":
                fraud_decision,

            "decision_source":
                "FRAUD_DECISION_ENGINE",

            "decision_reason":
                "EXISTING_FRAUD_MODEL_DECISION",

            "fraud_engine_considered":
                True,

            "transaction_review_policy_considered":
                True,

            "model_risk_score":
                risk_score
        }


    # ========================================================
    # SAFETY FALLBACK
    # ========================================================

    return {

        "final_decision":
            "REVIEW",

        "decision_source":
            "POLICY_ORCHESTRATOR",

        "decision_reason":
            "UNRESOLVED_POLICY_STATE",

        "fraud_engine_considered":
            False,

        "transaction_review_policy_considered":
            False
    }



# ------------------------------------------------------------
# PAYMENT INTELLIGENCE ORCHESTRATOR V3
#
# This later definition intentionally replaces the older
# analyze_payment_intelligence() contained earlier in the file.
# ------------------------------------------------------------

def analyze_payment_intelligence(
    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime
):


    # ========================================================
    # 1. MERCHANT RESTRICTION CONTROL
    # ========================================================

    merchant_control = (
        check_merchant_restriction(
            merchant_id
        )
    )


    # --------------------------------------------------------
    # RESTRICTED / UNKNOWN MERCHANT
    #
    # Stop before fraud/currency processing.
    # --------------------------------------------------------

    if (
        merchant_control[
            "platform_decision"
        ]
        !=
        "PROCEED"
    ):

        policy_result = (
            apply_policy_decision(
                merchant_control=
                    merchant_control,

                fraud_intelligence=
                    None,

                transaction_review_result=
                    None
            )
        )


        return {

            "transaction": {
                "customer_id":
                    customer_id,

                "amount":
                    amount,

                "merchant_id":
                    merchant_id,

                "country":
                    country,

                "channel":
                    channel,

                "entry_mode":
                    entry_mode,

                "device_id":
                    device_id,

                "transaction_datetime":
                    str(transaction_datetime)
            },

            "merchant_control":
                merchant_control,

            "transaction_review_policy":
                None,

            "policy_decision":
                policy_result,

            "platform_decision":
                policy_result[
                    "final_decision"
                ],

            "decision_source":
                policy_result[
                    "decision_source"
                ],

            "fraud_intelligence":
                None,

            "currency_intelligence":
                None
        }


    # ========================================================
    # 2. EXISTING FRAUD INTELLIGENCE
    # ========================================================

    fraud_intelligence = (
        analyze_transaction_end_to_end(

            customer_id=
                customer_id,

            amount=
                amount,

            merchant_id=
                merchant_id,

            country=
                country,

            channel=
                channel,

            entry_mode=
                entry_mode,

            device_id=
                device_id,

            transaction_datetime=
                transaction_datetime
        )
    )


    # ========================================================
    # 3. BUILD EXISTING BEHAVIORAL EVIDENCE
    #
    # This does NOT change XGBoost.
    # We need it only for the deterministic policy rules.
    # ========================================================

    transaction_features = (
        analyze_sandbox_transaction(

            customer_id=
                customer_id,

            amount=
                amount,

            merchant_id=
                merchant_id,

            country=
                country,

            channel=
                channel,

            entry_mode=
                entry_mode,

            device_id=
                device_id,

            transaction_datetime=
                transaction_datetime
        )
    )


    # ========================================================
    # 4. TRANSACTION REVIEW POLICY
    #
    # Rule A:
    # Amount >= 2X historical average
    #
    # Rule B:
    # Entirely new merchant for customer
    # ========================================================

    transaction_review_result = (
        evaluate_transaction_review_policy(
            transaction_features
        )
    )


    # ========================================================
    # 5. POLICY DECISION ORCHESTRATOR
    # ========================================================

    policy_result = (
        apply_policy_decision(

            merchant_control=
                merchant_control,

            fraud_intelligence=
                fraud_intelligence,

            transaction_review_result=
                transaction_review_result
        )
    )


    # ========================================================
    # 6. CURRENCY INTELLIGENCE
    #
    # Currency remains completely separate from fraud.
    # ========================================================

    currency_intelligence = (
        analyze_currency_recommendation_end_to_end(

            customer_id=
                customer_id,

            merchant_id=
                merchant_id,

            normalized_amount_inr=
                amount
        )
    )


    # ========================================================
    # 7. FINAL RESPONSE
    # ========================================================

    return {

        "transaction": {
            "customer_id":
                customer_id,

            "amount":
                amount,

            "merchant_id":
                merchant_id,

            "country":
                country,

            "channel":
                channel,

            "entry_mode":
                entry_mode,

            "device_id":
                device_id,

            "transaction_datetime":
                str(transaction_datetime)
        },

        "merchant_control":
            merchant_control,

        "transaction_review_policy":
            transaction_review_result,

        "policy_decision":
            policy_result,

        "platform_decision":
            policy_result[
                "final_decision"
            ],

        "decision_source":
            policy_result[
                "decision_source"
            ],

        "fraud_intelligence":
            fraud_intelligence,

        "currency_intelligence":
            currency_intelligence
    }



# ============================================================
# END POST-POLICY RECOVERY EXTENSION
# ============================================================



# ============================================================
# STEP-UP AUTHENTICATION RECOVERY EXTENSION
# Added after STEP 556
# ============================================================


# ------------------------------------------------------------
# REQUIRED STATE EXTENSION
# ------------------------------------------------------------

if "step_up_auth_policy" not in REQUIRED_STATE_KEYS:
    REQUIRED_STATE_KEYS = (
        list(REQUIRED_STATE_KEYS)
        +
        ["step_up_auth_policy"]
    )


# ------------------------------------------------------------
# STEP-UP AUTHENTICATION RESOLVER
# ------------------------------------------------------------

def resolve_step_up_authentication(
    current_decision,
    authentication_outcome=None
):

    # --------------------------------------------------------
    # 1. STEP-UP IS NOT REQUIRED
    # --------------------------------------------------------

    if (
        current_decision
        !=
        step_up_auth_policy[
            "trigger_decision"
        ]
    ):

        return {

            "step_up_required": False,

            "authentication_outcome": None,

            "final_decision":
                current_decision,

            "decision_source":
                "STEP_UP_NOT_REQUIRED",

            "decision_reason":
                "CURRENT_DECISION_DOES_NOT_REQUIRE_STEP_UP"
        }


    # --------------------------------------------------------
    # 2. REVIEW REQUIRES AN AUTHENTICATION OUTCOME
    # --------------------------------------------------------

    if authentication_outcome is None:

        return {

            "step_up_required": True,

            "authentication_outcome": None,

            "final_decision": "PENDING_AUTHENTICATION",

            "decision_source":
                "STEP_UP_AUTHENTICATION",

            "decision_reason":
                "AUTHENTICATION_OUTCOME_REQUIRED"
        }


    # --------------------------------------------------------
    # 3. NORMALIZE OUTCOME
    # --------------------------------------------------------

    authentication_outcome = (
        str(authentication_outcome)
        .strip()
        .upper()
    )


    # --------------------------------------------------------
    # 4. VALIDATE OUTCOME
    # --------------------------------------------------------

    valid_outcomes = (
        step_up_auth_policy[
            "outcomes"
        ]
    )


    if (
        authentication_outcome
        not in
        valid_outcomes
    ):

        raise ValueError(

            "authentication_outcome must be one of: "
            +
            ", ".join(
                valid_outcomes.keys()
            )
        )


    # --------------------------------------------------------
    # 5. RESOLVE AUTHENTICATION
    # --------------------------------------------------------

    outcome_policy = (
        valid_outcomes[
            authentication_outcome
        ]
    )


    return {

        "step_up_required": True,

        "authentication_outcome":
            authentication_outcome,

        "final_decision":
            outcome_policy[
                "final_decision"
            ],

        "decision_source":
            "STEP_UP_AUTHENTICATION",

        "decision_reason":
            outcome_policy[
                "reason"
            ]
    }



# ------------------------------------------------------------
# STEP-UP POLICY WRAPPER
# ------------------------------------------------------------

def apply_step_up_to_policy_result(
    policy_result,
    authentication_outcome=None
):

    # Existing decision produced by:
    # Merchant Control + XGBoost + Review Policy

    current_decision = (
        policy_result[
            "final_decision"
        ]
    )


    # Resolve Step-Up only when necessary

    step_up_result = (
        resolve_step_up_authentication(
            current_decision=
                current_decision,

            authentication_outcome=
                authentication_outcome
        )
    )


    # Preserve the decision BEFORE authentication.
    #
    # This gives us an audit trail such as:
    #
    # XGBoost       = APPROVE
    # Review Policy = REVIEW
    # Step-Up       = PASS
    # Final         = APPROVE

    return {

        "pre_auth_decision":
            current_decision,

        "pre_auth_decision_source":
            policy_result[
                "decision_source"
            ],

        "step_up_required":
            step_up_result[
                "step_up_required"
            ],

        "authentication_outcome":
            step_up_result[
                "authentication_outcome"
            ],

        "final_decision":
            step_up_result[
                "final_decision"
            ],

        "decision_source":
            step_up_result[
                "decision_source"
            ],

        "decision_reason":
            step_up_result[
                "decision_reason"
            ],

        "original_policy_result":
            policy_result
    }



# ------------------------------------------------------------
# PAYMENT INTELLIGENCE ORCHESTRATOR V4
#
# This later definition intentionally overrides
# the earlier V3 definition.
# ------------------------------------------------------------

def analyze_payment_intelligence(

    customer_id,
    amount,
    merchant_id,
    country,
    channel,
    entry_mode,
    device_id,
    transaction_datetime,

    authentication_outcome=None
):


    # --------------------------------------------------------
    # 1. MERCHANT CONTROL
    # --------------------------------------------------------

    merchant_control = (
        check_merchant_restriction(
            merchant_id
        )
    )


    # --------------------------------------------------------
    # 2. MERCHANT GATE SHORT-CIRCUIT
    #
    # Restricted / unknown merchants do not enter
    # fraud or currency intelligence.
    # --------------------------------------------------------

    if (
        merchant_control[
            "platform_decision"
        ]
        !=
        "PROCEED"
    ):

        policy_result = (
            apply_policy_decision(

                merchant_control=
                    merchant_control,

                fraud_intelligence=None,

                transaction_review_result=None
            )
        )


        step_up_result = (
            apply_step_up_to_policy_result(

                policy_result=
                    policy_result,

                authentication_outcome=
                    authentication_outcome
            )
        )


        return {

            "transaction": {

                "customer_id":
                    customer_id,

                "amount":
                    amount,

                "merchant_id":
                    merchant_id,

                "country":
                    country,

                "channel":
                    channel,

                "entry_mode":
                    entry_mode,

                "device_id":
                    device_id,

                "transaction_datetime":
                    transaction_datetime
            },


            "merchant_control":
                merchant_control,


            "transaction_review_policy":
                None,


            "policy_decision":
                policy_result,


            "step_up_authentication":
                step_up_result,


            # Decision before authentication
            "pre_auth_decision":
                policy_result[
                    "final_decision"
                ],


            # Ultimate platform decision
            "platform_decision":
                step_up_result[
                    "final_decision"
                ],


            "decision_source":
                step_up_result[
                    "decision_source"
                ],


            "fraud_intelligence":
                None,


            "currency_intelligence":
                None
        }


    # --------------------------------------------------------
    # 3. EXISTING FRAUD INTELLIGENCE
    #
    # Frozen XGBoost path.
    # --------------------------------------------------------

    fraud_intelligence = (
        analyze_transaction_end_to_end(

            customer_id=
                customer_id,

            amount=
                amount,

            merchant_id=
                merchant_id,

            country=
                country,

            channel=
                channel,

            entry_mode=
                entry_mode,

            device_id=
                device_id,

            transaction_datetime=
                transaction_datetime
        )
    )


    # --------------------------------------------------------
    # 4. REBUILD DETERMINISTIC TRANSACTION FEATURES
    #
    # Used only by Transaction Review Policy.
    # Not added to XGBoost.
    # --------------------------------------------------------

    transaction_features = (
        analyze_sandbox_transaction(

            customer_id=
                customer_id,

            amount=
                amount,

            merchant_id=
                merchant_id,

            country=
                country,

            channel=
                channel,

            entry_mode=
                entry_mode,

            device_id=
                device_id,

            transaction_datetime=
                transaction_datetime
        )
    )


    # --------------------------------------------------------
    # 5. TRANSACTION REVIEW POLICY
    # --------------------------------------------------------

    transaction_review_result = (
        evaluate_transaction_review_policy(
            transaction_features
        )
    )


    # --------------------------------------------------------
    # 6. POLICY DECISION ORCHESTRATOR
    #
    # Merchant + XGBoost + deterministic review rules.
    # --------------------------------------------------------

    policy_result = (
        apply_policy_decision(

            merchant_control=
                merchant_control,

            fraud_intelligence=
                fraud_intelligence,

            transaction_review_result=
                transaction_review_result
        )
    )


    # --------------------------------------------------------
    # 7. STEP-UP AUTHENTICATION
    #
    # REVIEW + no result  -> PENDING_AUTHENTICATION
    # REVIEW + PASS       -> APPROVE
    # REVIEW + FAIL       -> DECLINE
    # REVIEW + TIMEOUT    -> MANUAL_REVIEW
    #
    # APPROVE / DECLINE bypass Step-Up.
    # --------------------------------------------------------

    step_up_result = (
        apply_step_up_to_policy_result(

            policy_result=
                policy_result,

            authentication_outcome=
                authentication_outcome
        )
    )


    # --------------------------------------------------------
    # 8. CURRENCY INTELLIGENCE
    #
    # Remains completely separate from fraud decisioning.
    # --------------------------------------------------------

    currency_intelligence = (
        recovery
        .analyze_currency_recommendation_end_to_end(

            customer_id=
                customer_id,

            merchant_id=
                merchant_id,

            normalized_amount_inr=
                amount
        )
    )


    # --------------------------------------------------------
    # 9. FINAL RESPONSE
    # --------------------------------------------------------

    return {

        "transaction": {

            "customer_id":
                customer_id,

            "amount":
                amount,

            "merchant_id":
                merchant_id,

            "country":
                country,

            "channel":
                channel,

            "entry_mode":
                entry_mode,

            "device_id":
                device_id,

            "transaction_datetime":
                transaction_datetime
        },


        "merchant_control":
            merchant_control,


        "transaction_review_policy":
            transaction_review_result,


        "policy_decision":
            policy_result,


        "step_up_authentication":
            step_up_result,


        # Useful audit field:
        # decision BEFORE authentication.
        "pre_auth_decision":
            policy_result[
                "final_decision"
            ],


        # Ultimate platform decision AFTER authentication.
        "platform_decision":
            step_up_result[
                "final_decision"
            ],


        "decision_source":
            step_up_result[
                "decision_source"
            ],


        "fraud_intelligence":
            fraud_intelligence,


        "currency_intelligence":
            currency_intelligence
    }



# ============================================================
# END STEP-UP AUTHENTICATION RECOVERY EXTENSION
# ============================================================



# ============================================================
# ECONOMIC DECISIONING + CALIBRATION RECOVERY EXTENSION
# Added after STEP 568
# ============================================================


# ------------------------------------------------------------
# REQUIRED STATE EXTENSION
# ------------------------------------------------------------

additional_required_state_keys = [

    "economic_decision_policy",

    "calibration_positions",
    "validation_positions",

    "X_calibration",
    "y_calibration",

    "X_calibration_validation",
    "y_calibration_validation",

    "fraud_probability_calibrator",

    "calibration_raw_scores",
    "validation_raw_scores",
    "validation_calibrated_probabilities",

    "benchmark_calibrated_probability",

    "raw_brier_score",
    "calibrated_brier_score",
    "brier_improvement_pct"
]


for _key in additional_required_state_keys:

    if _key not in REQUIRED_STATE_KEYS:

        REQUIRED_STATE_KEYS = (
            list(REQUIRED_STATE_KEYS)
            +
            [_key]
        )


# ------------------------------------------------------------
# FALSE-DECLINE ECONOMIC COST
# ------------------------------------------------------------

def calculate_false_decline_cost(
    transaction_amount
):

    policy = (
        economic_decision_policy[
            "false_decline"
        ]
    )


    # --------------------------------------------------------
    # 1. CUSTOMER SERVICE / OPERATIONS COST
    # --------------------------------------------------------

    service_cost = (
        policy[
            "service_cost_inr"
        ]
    )


    # --------------------------------------------------------
    # 2. LOST TRANSACTION REVENUE
    # --------------------------------------------------------

    lost_transaction_revenue = (
        transaction_amount
        *
        policy[
            "transaction_revenue_rate"
        ]
    )


    # --------------------------------------------------------
    # 3. EXPECTED CUSTOMER ATTRITION / FRICTION COST
    # --------------------------------------------------------

    expected_attrition_cost = (

        policy[
            "customer_abandonment_probability"
        ]

        *

        policy[
            "customer_attrition_cost_inr"
        ]
    )


    # --------------------------------------------------------
    # 4. TOTAL FALSE-DECLINE COST
    # --------------------------------------------------------

    total_false_decline_cost = (

        service_cost
        +
        lost_transaction_revenue
        +
        expected_attrition_cost
    )


    return {

        "transaction_amount_inr":
            float(transaction_amount),

        "service_cost_inr":
            float(service_cost),

        "lost_transaction_revenue_inr":
            float(lost_transaction_revenue),

        "expected_attrition_cost_inr":
            float(expected_attrition_cost),

        "total_false_decline_cost_inr":
            float(total_false_decline_cost)
    }



# ------------------------------------------------------------
# FRAUD LOSS ECONOMIC COST
# ------------------------------------------------------------

def calculate_fraud_loss(
    transaction_amount
):

    policy = (
        economic_decision_policy[
            "fraud_loss"
        ]
    )


    # --------------------------------------------------------
    # 1. LOSS-GIVEN-FRAUD ASSUMPTION
    # --------------------------------------------------------

    loss_given_fraud_rate = (
        policy[
            "loss_given_fraud_rate"
        ]
    )


    # --------------------------------------------------------
    # 2. FRAUD LOSS
    # --------------------------------------------------------

    fraud_loss = (
        transaction_amount
        *
        loss_given_fraud_rate
    )


    # --------------------------------------------------------
    # 3. RETURN TRANSPARENT BREAKDOWN
    # --------------------------------------------------------

    return {

        "transaction_amount_inr":
            float(transaction_amount),

        "loss_given_fraud_rate":
            float(loss_given_fraud_rate),

        "fraud_loss_inr":
            float(fraud_loss)
    }



# ============================================================
# END ECONOMIC DECISIONING + CALIBRATION EXTENSION
# ============================================================
