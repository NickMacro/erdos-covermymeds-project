import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

CODE_TRANSLATION = {70: "the drug is not on the formulary and not covered by the plan",
                    75: "the drug is on the formulary but requires prior authorization",
                    76: "the plan limitations have been exceeded"}


@st.cache
def load_data(claims_path, pa_path, bridge_path):
    claims_df = pd.read_csv(claims_path).fillna(0)
    claims_df.loc[:, 'reject_code'] = claims_df['reject_code'].astype(int)

    pa_df = pd.read_csv(pa_path)
    bridge_df = pd.read_csv(bridge_path)

    pa_combined_df = bridge_df.merge(claims_df, on='dim_claim_id').merge(pa_df, on='dim_pa_id').drop(columns=['dim_claim_id', 'dim_pa_id', 'dim_date_id'])
    return claims_df, pa_combined_df


def train_claims_model(data_df):
    claims_X = data_df[['bin', 'drug']].values
    claims_y = data_df['pharmacy_claim_approved'].values

    bin_le = LabelEncoder()
    claims_X[:, 0] = bin_le.fit_transform(claims_X[:, 0].astype(int))

    drug_le = LabelEncoder()
    claims_X[:, 1] = drug_le.fit_transform(claims_X[:, 1])

    claims_model = DecisionTreeClassifier(random_state=42)
    claims_model.fit(claims_X, claims_y)
    return claims_model, bin_le, drug_le


def train_pa_model(data_df, bin_le, drug_le):
    pa_X = data_df[['bin', 'drug', 'correct_diagnosis', 'tried_and_failed', 'contraindication']].values
    pa_y = data_df['pa_approved'].values

    pa_X[:, 0] = bin_le.fit_transform(pa_X[:, 0].astype(int))
    pa_X[:, 1] = drug_le.fit_transform(pa_X[:, 1])

    pa_model = DecisionTreeClassifier(random_state=42)
    pa_model.fit(pa_X, pa_y)
    return pa_model


def render_sidebar(page_names):
    state = st.sidebar.radio('Table of Contents', page_names)
    st.sidebar.write("This project was performed as part of the Erdős Institute's Spring 2021 Data Science Bootcamp.")
    st.sidebar.write("The data were provided by CoverMyMeds.")
    st.sidebar.write("Created by Nick Macro, Sandrine Müller, and Tomas Kasza.")
    return state


def render_introduction_page():
    st.header("Introduction")
    st.write("Patients can encounter difficulties with insurance coverage of specific drugs, depending on the insurance payer.")


def render_data_page():
    st.header("Data")
    st.write("The data consists of ~1.3 million claims records with ~500k prior authorizations filed.")
    claims_df, pa_combined_df = load_data("./data/processed/dim_claims_train.csv",
                                         "./data/processed/dim_pa_train.csv",
                                         "./data/processed/bridge_train.csv")

    payer = int(st.selectbox('Enter payer ID.', sorted(claims_df['bin'].unique())))

    claim_view = claims_df.loc[claims_df['bin'] == payer]
    drug_coverage = pd.DataFrame()
    drug_coverage.loc[:, 'Claims Submitted'] = claim_view.groupby('drug')['pharmacy_claim_approved'].count()
    drug_coverage.loc[:, 'Claim Approval %'] = round(100 * claim_view.groupby('drug')['pharmacy_claim_approved'].mean())
    drug_coverage.loc[:, 'Drug'] = drug_coverage.index
    drug_coverage.index = ['' for _ in range(len(drug_coverage))]
    drug_coverage = drug_coverage[['Drug', 'Claims Submitted', 'Claim Approval %']]
    st.table(drug_coverage)

    pa_view = pa_combined_df.loc[pa_combined_df['bin'] == payer]
    for drug in pa_view['drug'].unique():
        reject_code = pa_view.loc[pa_view['drug'] == drug]['reject_code'].unique()[0]
        drug_rejection = round(100 * (1 - claim_view.loc[claim_view['drug'] == drug]['pharmacy_claim_approved'].mean()))
        text = f"If rejected ({drug_rejection}%), drug {drug} is always rejected because {CODE_TRANSLATION[reject_code]} (code {reject_code})."
        st.write(text)

    drug = st.selectbox('Enter drug name.', sorted(claims_df['drug'].unique()))

    pa_names = {'contraindication': 'Contraindication',
                'tried_and_failed': 'Failed Generic',
                'correct_diagnosis': 'Correct Diagnosis'}

    pa_view = pa_combined_df.loc[(pa_combined_df['bin'] == payer) & (pa_combined_df['drug'] == drug)]
    rejection_table = pd.DataFrame()
    rejection_table.loc[:, 'pa_ammount'] = pa_view.groupby(list(pa_names.keys()))['pa_approved'].count()
    rejection_table.loc[:, 'pa_rate'] = pa_view.groupby(list(pa_names.keys()))['pa_approved'].mean()
    rejection_table = rejection_table.sort_values('pa_rate', ascending=False)

    rejection_display = pd.DataFrame()
    for key, value in pa_names.items():
        rejection_display.loc[:, value] = rejection_table.index.get_level_values(key).astype(bool)
    rejection_display.loc[:, 'PAs Submitted'] = rejection_table['pa_ammount'].values
    rejection_display.loc[:, 'PA Approval %'] = 100 * rejection_table['pa_rate'].values
    rejection_display.loc[:, 'PA Approval %'] = rejection_display['PA Approval %'].round()
    rejection_display.index = ['' for _ in range(len(rejection_display))]
    st.table(rejection_display)

    for name in pa_names.keys():
        true_rate = pa_view.loc[pa_view[name] == 1]['pa_approved'].mean()
        false_rate = pa_view.loc[pa_view[name] == 0]['pa_approved'].mean()

        rate_delta = int(round(100 * (true_rate - false_rate), 0))
        if rate_delta >= 0:
            change = 'increases'
        else:
            change = 'decreases'
        st.write(f"{pa_names[name]} {change} the PA Approval % by {abs(rate_delta)}%.")



    



def render_model_page():
    st.header("Model")
    st.write("A decision tree model was found to be the best performing model for both:")
    st.write("1. Predicting if a claim will be approved by the payer.")
    st.write("2. Predicting if a prior authorization will be approved by the payer.")


def render_prototype_page():
    st.header("Prototype")
    st.write("This model can be applied to perform predictions on future claims and prior authorizations.")

    claims_df, pa_combined_df = load_data("./data/processed/dim_claims_train.csv",
                                          "./data/processed/dim_pa_train.csv",
                                          "./data/processed/bridge_train.csv")

    payer = int(st.selectbox('Enter payer ID.', sorted(claims_df['bin'].unique())))
    drug = st.selectbox('Enter drug name.', sorted(claims_df['drug'].unique()))

    claims_model, bin_le, drug_le = train_claims_model(claims_df)
    pa_model = train_pa_model(pa_combined_df, bin_le, drug_le)

    payer = bin_le.transform(np.array([payer]))[0]
    drug = drug_le.transform(np.array([drug]))[0]

    claim_pred = claims_model.predict(np.array([[payer, drug]]))[0]
    claim_prob = claims_model.predict_proba(np.array([[payer, drug]]))[0, 1]

    chance = {0: 'unlikely', 1: 'likely'}

    st.write("The prescription is **" + chance[claim_pred] + "** (" + str(int(100 * claim_prob)) + "%) to be approved.")

    if not claim_pred:
        st.write('The prescription is likely to require a prior authorization. Fill out the following details to determine if the prior authorization is likely to be accepted.')
        correct_diagnosis = st.checkbox('Drug is appropriate for the diagnosis.')
        tried_and_failed = st.checkbox('Tried and failed generic alternative.')
        contraindication = st.checkbox('Contraindication present for the drug.')

        pa_pred = pa_model.predict(np.array([[payer, drug, correct_diagnosis, tried_and_failed, contraindication]]))[0]
        pa_prob = pa_model.predict_proba(np.array([[payer, drug, correct_diagnosis, tried_and_failed, contraindication]]))[0, 1]
        st.write("The prior authorization is **" + chance[pa_pred] + "** (" + str(int(100 * pa_prob)) + "%) to be approved.")


pages = {'Introduction': render_introduction_page,
         'Data': render_data_page,
         'Model': render_model_page,
         'Prototype': render_prototype_page}

st.title("Willow: A Medicinal Tree")
state = render_sidebar(list(pages.keys()))
pages[state]()
