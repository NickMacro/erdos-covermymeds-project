import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


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
    st.write("The data consists of ~1.3 million claims records with ~500k PAs filed.")


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

    bin = int(st.selectbox('Enter payer ID.', sorted(claims_df['bin'].unique())))
    drug = st.selectbox('Enter drug name.', sorted(claims_df['drug'].unique()))

    claims_model, bin_le, drug_le = train_claims_model(claims_df)
    pa_model = train_pa_model(pa_combined_df, bin_le, drug_le)

    bin = bin_le.transform(np.array([bin]))[0]
    drug = drug_le.transform(np.array([drug]))[0]

    claim_pred = claims_model.predict(np.array([[bin, drug]]))[0]
    claim_prob = claims_model.predict_proba(np.array([[bin, drug]]))[0, 1]

    chance = {0: 'unlikely', 1: 'likely'}

    st.write("The prescription is **" + chance[claim_pred] + "** (" + str(int(100 * claim_prob)) + "%) to be approved.")

    if not claim_pred:
        st.write('The prescription is likely to require a PA. Fill out the following details to determine if the PA is likely to be accepted.')
        correct_diagnosis = st.checkbox('Drug is appropriate for the diagnosis.')
        tried_and_failed = st.checkbox('Tried and failed generic alternative.')
        contraindication = st.checkbox('Contraindication present for the drug.')

        pa_pred = pa_model.predict(np.array([[bin, drug, correct_diagnosis, tried_and_failed, contraindication]]))[0]
        pa_prob = pa_model.predict_proba(np.array([[bin, drug, correct_diagnosis, tried_and_failed, contraindication]]))[0, 1]
        st.write("The PA is **" + chance[pa_pred] + "** (" + str(int(100 * pa_prob)) + "%) to be approved.")


pages = {'Introduction': render_introduction_page,
         'Data': render_data_page,
         'Model': render_model_page,
         'Prototype': render_prototype_page}

st.title("Willow: A Medicinal Tree")
state = render_sidebar(list(pages.keys()))
pages[state]()
