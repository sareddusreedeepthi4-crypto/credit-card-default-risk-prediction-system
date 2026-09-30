import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Credit Card Defaulter Prediction",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       PAGE
       ======================================================== */

    .stApp {
        background-color: #d0d0d0;
    }

    .block-container {
        max-width: 1150px !important;
        padding-top: 18px !important;
        padding-bottom: 10px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    /* Hide Streamlit UI */
    [data-testid="stSidebar"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden !important;
    }

    header {
        visibility: hidden !important;
        height: 0 !important;
    }

    footer {
        visibility: hidden !important;
    }


    /* ========================================================
       TITLE
       ======================================================== */

    .app-title {
        width: 430px;
        margin: 0 auto 12px auto;
        padding: 8px 12px;

        border: 2px solid #333;
        background-color: #d0d0d0;

        text-align: center;

        font-family: Georgia, "Times New Roman", serif;
        font-size: 21px;
        font-weight: bold;
        line-height: 25px;

        color: #111;
        box-sizing: border-box;
    }


    /* ========================================================
       MAIN BOXES
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #d0d0d0 !important;
        border: 1px solid #999 !important;
        border-radius: 3px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 10px 12px !important;
    }


    /* ========================================================
       HEADINGS
       ======================================================== */

    .section-title {
        margin: 0 0 7px 0;

        font-family: Georgia, "Times New Roman", serif;
        font-size: 15px;
        font-weight: bold;
        line-height: 18px;

        color: #111;
    }

    .field-title {
        margin: 4px 0 3px 0;

        font-family: Georgia, "Times New Roman", serif;
        font-size: 13px;
        font-weight: bold;
        line-height: 16px;

        color: #111;
    }

    .description {
        margin: 0 0 4px 0;

        font-family: Georgia, "Times New Roman", serif;
        font-size: 11px;
        line-height: 14px;

        color: #222;
    }

    .month-label {
        margin: 0 0 2px 0;

        font-family: Georgia, "Times New Roman", serif;
        font-size: 11px;
        line-height: 14px;

        color: #111;
        white-space: nowrap;
    }


    /* ========================================================
       RADIO BUTTONS
       ======================================================== */

    div[data-testid="stRadio"] {
        margin: 0 0 7px 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stRadio"] > label {
        display: none !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;

        align-items: center !important;

        gap: 3px 9px !important;

        min-height: 18px !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        margin: 0 !important;
        padding: 0 !important;

        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 12px !important;
        line-height: 16px !important;

        color: #222 !important;
    }


    /* ========================================================
       TEXT INPUTS
       ======================================================== */

    div[data-testid="stTextInput"] {
        margin: 0 0 5px 0 !important;
        padding: 0 !important;
    }

    div[data-testid="stTextInput"] > label {
        display: none !important;
    }

    div[data-testid="stTextInput"] input {
        width: 100% !important;
        height: 27px !important;
        min-height: 27px !important;

        padding: 3px 6px !important;

        box-sizing: border-box !important;

        border: 1px solid #888 !important;
        border-radius: 0 !important;

        background-color: #ffffff !important;
        color: #111111 !important;

        font-family: Arial, sans-serif !important;
        font-size: 11px !important;
        line-height: 18px !important;
    }


    /* ========================================================
       REMOVE ONLY SMALL DEFAULT SPACING
       DO NOT TOUCH STREAMLIT VERTICAL BLOCK HEIGHTS
       ======================================================== */

    div[data-testid="stElementContainer"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }


    /* ========================================================
       PREDICT BUTTON
       ======================================================== */

    .button-area {
        text-align: center;
        margin-top: 8px;
    }

    div.stButton {
        display: flex !important;
        justify-content: center !important;
        margin: 0 !important;
    }

    div.stButton > button {
        width: 75px !important;
        min-width: 75px !important;

        height: 30px !important;
        min-height: 30px !important;

        padding: 2px 8px !important;

        border: 1px solid #888 !important;
        border-radius: 0 !important;

        background-color: #eeeeee !important;
        color: #111111 !important;

        font-family: Georgia, "Times New Roman", serif !important;
        font-size: 12px !important;
        font-weight: normal !important;
    }


    /* ========================================================
       RESULT
       ======================================================== */

    .prediction {
        text-align: center;

        margin-top: 6px;

        font-family: Georgia, "Times New Roman", serif;
        font-size: 18px;
        font-weight: bold;
    }

    .prediction-default {
        color: #990000;
    }

    .prediction-safe {
        color: #006600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("model.pkl was not found. Keep model.pkl in the same folder as app.py.")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

except Exception as exc:
    st.error("The model could not be loaded.")
    st.code(str(exc))
    st.stop()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="app-title">Credit Card Defaulter Prediction</div>',
    unsafe_allow_html=True,
)


# ============================================================
# MAIN TWO-COLUMN LAYOUT
# ============================================================

left_col, right_col = st.columns(
    [1, 1],
    gap="small"
)


# ============================================================
# DEMOGRAPHIC DATA
# ============================================================

with left_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Demographic data:</div>',
            unsafe_allow_html=True,
        )


        # ----------------------------------------------------
        # GENDER
        # ----------------------------------------------------

        st.markdown(
            '<div class="field-title">Gender:</div>',
            unsafe_allow_html=True,
        )

        gender = st.radio(
            "Gender",
            ["Male", "Female"],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )


        # ----------------------------------------------------
        # EDUCATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="field-title">Education:</div>',
            unsafe_allow_html=True,
        )

        education = st.radio(
            "Education",
            [
                "Graduate School",
                "University",
                "High School",
                "Others",
                "Unknown",
            ],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )


        # ----------------------------------------------------
        # MARITAL STATUS
        # ----------------------------------------------------

        st.markdown(
            '<div class="field-title">Marital Status:</div>',
            unsafe_allow_html=True,
        )

        marital_status = st.radio(
            "Marital Status",
            [
                "Married",
                "Single",
                "Others",
            ],
            index=0,
            horizontal=True,
            label_visibility="collapsed",
        )


        # ----------------------------------------------------
        # AGE
        # ----------------------------------------------------

        st.markdown(
            '<div class="field-title">Age:</div>',
            unsafe_allow_html=True,
        )

        age_text = st.text_input(
            "Age",
            value="25",
            label_visibility="collapsed",
        )


        # ----------------------------------------------------
        # LIMIT BALANCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="field-title">Limit Balance:</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="description">'
            'Amount of given credit in dollar '
            '(includes individual and family/supplementary credit)'
            '</div>',
            unsafe_allow_html=True,
        )

        limit_balance_text = st.text_input(
            "Limit Balance",
            value="50000",
            label_visibility="collapsed",
        )


# ============================================================
# BEHAVIORAL DATA
# ============================================================

with right_col:

    with st.container(border=True):

        st.markdown(
            '<div class="section-title">Behavioral data:</div>',
            unsafe_allow_html=True,
        )


        # ====================================================
        # REPAYMENT STATUS
        # ====================================================

        st.markdown(
            '<div class="field-title">Repayment Status:</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="description">'
            '(-1—pay duly, 1—one month delay, 2—two months delay, '
            '... 9—delay for nine months and above)'
            '</div>',
            unsafe_allow_html=True,
        )

        repayment = {}

        repayment_fields = [
            ("April", "repayment_april"),
            ("May", "repayment_may"),
            ("June", "repayment_june"),
            ("July", "repayment_july"),
            ("August", "repayment_august"),
            ("September", "repayment_september"),
        ]

        repayment_cols = st.columns(6, gap="small")

        for col, (month, key) in zip(
            repayment_cols,
            repayment_fields
        ):

            with col:

                st.markdown(
                    f'<div class="month-label">{month}</div>',
                    unsafe_allow_html=True,
                )

                repayment[key] = st.text_input(
                    key,
                    value="0",
                    label_visibility="collapsed",
                )


        # ====================================================
        # BILL AMOUNTS
        # ====================================================

        st.markdown(
            '<div class="field-title">'
            'Bill Amounts: '
            '<span style="font-weight: normal;">'
            'Amount of bill statements (in dollar)'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        bill = {}

        bill_fields = [
            ("April", "bill_april"),
            ("May", "bill_may"),
            ("June", "bill_june"),
            ("July", "bill_july"),
            ("August", "bill_august"),
            ("September", "bill_september"),
        ]

        bill_cols = st.columns(3, gap="small")

        for col, (month, key) in zip(
            bill_cols,
            bill_fields[:3]
        ):

            with col:

                st.markdown(
                    f'<div class="month-label">{month}</div>',
                    unsafe_allow_html=True,
                )

                bill[key] = st.text_input(
                    key,
                    value="0",
                    label_visibility="collapsed",
                )


        bill_cols = st.columns(3, gap="small")

        for col, (month, key) in zip(
            bill_cols,
            bill_fields[3:]
        ):

            with col:

                st.markdown(
                    f'<div class="month-label">{month}</div>',
                    unsafe_allow_html=True,
                )

                bill[key] = st.text_input(
                    key,
                    value="0",
                    label_visibility="collapsed",
                )


        # ====================================================
        # PREVIOUS PAYMENTS
        # ====================================================

        st.markdown(
            '<div class="field-title">'
            'Previous Payments: '
            '<span style="font-weight: normal;">'
            'Amount of previous payments (in dollar)'
            '</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        payment = {}

        payment_fields = [
            ("April", "payment_april"),
            ("May", "payment_may"),
            ("June", "payment_june"),
            ("July", "payment_july"),
            ("August", "payment_august"),
            ("September", "payment_september"),
        ]

        payment_cols = st.columns(3, gap="small")

        for col, (month, key) in zip(
            payment_cols,
            payment_fields[:3]
        ):

            with col:

                st.markdown(
                    f'<div class="month-label">{month}</div>',
                    unsafe_allow_html=True,
                )

                payment[key] = st.text_input(
                    key,
                    value="0",
                    label_visibility="collapsed",
                )


        payment_cols = st.columns(3, gap="small")

        for col, (month, key) in zip(
            payment_cols,
            payment_fields[3:]
        ):

            with col:

                st.markdown(
                    f'<div class="month-label">{month}</div>',
                    unsafe_allow_html=True,
                )

                payment[key] = st.text_input(
                    key,
                    value="0",
                    label_visibility="collapsed",
                )


# ============================================================
# PREDICT BUTTON
# ============================================================

st.markdown(
    '<div class="button-area"></div>',
    unsafe_allow_html=True,
)

predict_clicked = st.button("Predict")


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    # --------------------------------------------------------
    # VALIDATE INPUTS
    # --------------------------------------------------------

    try:

        age = int(float(age_text))
        limit_balance = float(limit_balance_text)

        pay_april = int(float(repayment["repayment_april"]))
        pay_may = int(float(repayment["repayment_may"]))
        pay_june = int(float(repayment["repayment_june"]))
        pay_july = int(float(repayment["repayment_july"]))
        pay_august = int(float(repayment["repayment_august"]))
        pay_september = int(float(repayment["repayment_september"]))

        bill_april = float(bill["bill_april"])
        bill_may = float(bill["bill_may"])
        bill_june = float(bill["bill_june"])
        bill_july = float(bill["bill_july"])
        bill_august = float(bill["bill_august"])
        bill_september = float(bill["bill_september"])

        payment_april = float(payment["payment_april"])
        payment_may = float(payment["payment_may"])
        payment_june = float(payment["payment_june"])
        payment_july = float(payment["payment_july"])
        payment_august = float(payment["payment_august"])
        payment_september = float(payment["payment_september"])

    except (ValueError, TypeError):

        st.error(
            "Please enter valid numeric values in all input fields."
        )

        st.stop()


    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not 21 <= age <= 80:

        st.error(
            "Age must be between 21 and 80."
        )

        st.stop()


    repayment_values_ui = [
        pay_april,
        pay_may,
        pay_june,
        pay_july,
        pay_august,
        pay_september,
    ]

    if any(
        value < -2 or value > 9
        for value in repayment_values_ui
    ):

        st.error(
            "Repayment Status values must be between -2 and 9."
        )

        st.stop()


    # --------------------------------------------------------
    # MODEL CATEGORY VALUES
    # --------------------------------------------------------

    sex = gender

    education_model = education

    marriage_model = (
        "Other"
        if marital_status == "Others"
        else marital_status
    )


    # --------------------------------------------------------
    # ORIGINAL UCI MONTH MAPPING
    #
    # April      -> PAY_6 / BILL_AMT6 / PAY_AMT6
    # May        -> PAY_5 / BILL_AMT5 / PAY_AMT5
    # June       -> PAY_4 / BILL_AMT4 / PAY_AMT4
    # July       -> PAY_3 / BILL_AMT3 / PAY_AMT3
    # August     -> PAY_2 / BILL_AMT2 / PAY_AMT2
    # September  -> PAY_0 / BILL_AMT1 / PAY_AMT1
    # --------------------------------------------------------

    pay_6 = pay_april
    pay_5 = pay_may
    pay_4 = pay_june
    pay_3 = pay_july
    pay_2 = pay_august
    pay_0 = pay_september

    bill_6 = bill_april
    bill_5 = bill_may
    bill_4 = bill_june
    bill_3 = bill_july
    bill_2 = bill_august
    bill_1 = bill_september

    pay_amt6 = payment_april
    pay_amt5 = payment_may
    pay_amt4 = payment_june
    pay_amt3 = payment_july
    pay_amt2 = payment_august
    pay_amt1 = payment_september


    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    repayment_values = [
        pay_0,
        pay_2,
        pay_3,
        pay_4,
        pay_5,
        pay_6,
    ]


    avg_bill = np.mean(
        [
            bill_1,
            bill_2,
            bill_3,
            bill_4,
            bill_5,
            bill_6,
        ]
    )


    delayed_months = sum(
        value > 0
        for value in repayment_values
    )


    max_repayment_delay = max(
        repayment_values
    )


    total_bill = sum(
        [
            bill_1,
            bill_2,
            bill_3,
            bill_4,
            bill_5,
            bill_6,
        ]
    )


    total_payment = sum(
        [
            pay_amt1,
            pay_amt2,
            pay_amt3,
            pay_amt4,
            pay_amt5,
            pay_amt6,
        ]
    )


    payment_bill_ratio = (
        total_payment / total_bill
        if total_bill != 0
        else 0
    )


    # --------------------------------------------------------
    # AGE GROUP
    # --------------------------------------------------------

    age_group = pd.cut(
        pd.Series([age]),
        bins=[20, 30, 40, 50, 60, 80],
        labels=[
            "21-30",
            "31-40",
            "41-50",
            "51-60",
            "61-80",
        ],
    ).iloc[0]


    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        [
            {
                "LIMIT_BAL": limit_balance,

                "SEX": sex,

                "EDUCATION": education_model,

                "MARRIAGE": marriage_model,

                "AGE": age,

                "PAY_0": pay_0,
                "PAY_2": pay_2,
                "PAY_3": pay_3,
                "PAY_4": pay_4,
                "PAY_5": pay_5,
                "PAY_6": pay_6,

                "BILL_AMT1": bill_1,
                "BILL_AMT2": bill_2,
                "BILL_AMT3": bill_3,
                "BILL_AMT4": bill_4,
                "BILL_AMT5": bill_5,
                "BILL_AMT6": bill_6,

                "PAY_AMT1": pay_amt1,
                "PAY_AMT2": pay_amt2,
                "PAY_AMT3": pay_amt3,
                "PAY_AMT4": pay_amt4,
                "PAY_AMT5": pay_amt5,
                "PAY_AMT6": pay_amt6,

                "AVG_BILL": avg_bill,

                "AGE_GROUP": age_group,

                "DELAYED_MONTHS": delayed_months,

                "MAX_REPAYMENT_DELAY": max_repayment_delay,

                "PAYMENT_BILL_RATIO": payment_bill_ratio,
            }
        ]
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        prediction = int(
            model.predict(input_data)[0]
        )

        if prediction == 1:

            st.markdown(
                '<div class="prediction prediction-default">'
                'Default'
                '</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="prediction prediction-safe">'
                'Not Default'
                '</div>',
                unsafe_allow_html=True,
            )

    except Exception as exc:

        st.error(
            "Prediction could not be completed."
        )

        st.code(str(exc))