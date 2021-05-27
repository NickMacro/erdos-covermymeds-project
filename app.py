import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from joblib import load

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


def render_sidebar(page_names):
    state = st.sidebar.radio('Table of Contents', page_names)
    st.sidebar.write("This project was performed as part of the Erdős Institute's Spring 2021 Data Science Bootcamp.")
    st.sidebar.write("The data were provided by CoverMyMeds.")
    st.sidebar.write("Created by Nick Macro, Sandrine Müller, and Tomas Kasza.")
    return state


def render_introduction_page():
    st.header("Introduction")
    st.write("Patients can encounter difficulties with insurance coverage of specific drugs, depending on the insurance payer. We are hoping to solve the following problems: 1. Can the approval of a PA be predicted using claim and PA data? If so, what factors tend to influence the approval of a PA? 2. Can a claim’s rejection be predicted using claim and PA data? If so, what factors tend to influence the rejection of a claim?")


def render_data_page():
    st.header("Data")
    st.write("The data consists of four tables of data pertaining to patients’ access to drugs and prior authorizations (PA) due to rejection by the payer. There are ~1.3 million claims across three years from January 1, 2017 to December 31, 2019. The pharmacy claim-level data provided is for three drugs (A, B, C) and four payers (417380, 417614, 417740, 999001). There are ~556k rejected claims that require a PA. There are three rejection codes provided; '70' for a drug that is not covered by the plan and not on formulary, '75' for a drug on the formulary that does not have preferred status and requires a PA, and “76” for a drug that is covered but the plan limitations have been exceeded. The PA data contains four binary categories indicating whether the patient has the correct diagnosis (80% of PAs), has tried and failed a generic alternative (50% of PAs), if the patient has an associated contraindication (20% of PAs), and whether the PA was approved (73% of PAs).")
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


def bokeh_image_viewer(left, right, top, bottom, img_url):
    f = figure(x_range=(left, right), 
            y_range=(bottom, top),
            toolbar_location='below',
            tools='pan, wheel_zoom, zoom_in, zoom_out, reset, save')
    f.xgrid.visible = False
    f.ygrid.visible = False
    f.axis.visible = False
    f.image_url(url=[img_url], x=0, y=1, w=1, h=1)
    return f


def render_model_page():

    img_urls = {'claim_approval': 'https://github.com/NickMacro/erdos-covermymeds-project/raw/main/models/saved-model-figures/claim-approval-tree.png',
                'reject_code': 'https://github.com/NickMacro/erdos-covermymeds-project/raw/main/models/saved-model-figures/reject-code-tree.png',
                'pa_approval': 'https://github.com/NickMacro/erdos-covermymeds-project/raw/main/models/saved-model-figures/pa-approval-tree.png'}

    st.header("Model")
    st.write("A decision tree model was found to be the best performing model for both:")
    st.write("1. Predicting if a claim will be approved by the payer.")
    st.write("2. Predicting if a prior authorization will be approved by the payer.")

    interactive_plots = st.checkbox('Interactive Plots')

    st.subheader('Claim Approval Decision Tree')
    if interactive_plots:
        left = 0.5
        right = left + 0.25
        bottom = 0.5
        top = bottom + 0.5
        f = bokeh_image_viewer(left, right, top, bottom, img_urls['claim_approval'])
        st.bokeh_chart(f, use_container_width=True)
    else:
        st.image(img_urls['claim_approval'], use_column_width ='always')

    st.subheader('Rejection Code Decision Tree')
    if interactive_plots:
        left = 0.55
        right = left + 0.25
        bottom = 0.5
        top = bottom + 0.5
        f = bokeh_image_viewer(left, right, top, bottom, img_urls['reject_code'])
        st.bokeh_chart(f, use_container_width=True)
    else:
        st.image(img_urls['reject_code'], use_column_width ='always')
    
    st.subheader('Prior Authorization Approval Decision Tree')
    if interactive_plots:
        left = 0.51
        right = left + 0.1
        bottom = 0.2
        top = bottom + 0.7
        f = bokeh_image_viewer(left, right, top, bottom, img_urls['pa_approval'])
        st.bokeh_chart(f, use_container_width=True)
    else:
        st.image(img_urls['pa_approval'], use_column_width ='always')

def render_prototype_page():
    st.header("Prototype")
    st.write("This model can be applied to perform predictions on future claims and prior authorizations.")

    claims_df, pa_combined_df = load_data("./data/processed/dim_claims_train.csv",
                                          "./data/processed/dim_pa_train.csv",
                                          "./data/processed/bridge_train.csv")

    payer = int(st.selectbox('Enter payer ID.', sorted(claims_df['bin'].unique())))
    drug = st.selectbox('Enter drug name.', sorted(claims_df['drug'].unique()))
    user_claims_X = pd.DataFrame({'bin': [payer], 'drug': [drug]})

    claims_pipe = load(r"./models/saved-models/decision-tree-claim-approval.joblib")
    claim_pred = claims_pipe.predict(user_claims_X)[0]
    claim_prob = claims_pipe.predict_proba(user_claims_X)[0, 1]

    chance = {0: 'unlikely', 1: 'likely'}

    st.write("The prescription is **" + chance[claim_pred] + "** (" + str(int(100 * claim_prob)) + "%) to be approved.")

    if not claim_pred:
        reject_pipe = load(r"./models/saved-models/decision-tree-reject-code.joblib")
        reject_pred = reject_pipe.predict(user_claims_X)[0]

        st.write(f'The prescription is likely to require a prior authorization because {CODE_TRANSLATION[reject_pred]}. Fill out the following details to determine if the prior authorization is likely to be accepted.')
        correct_diagnosis = int(st.checkbox('Drug is appropriate for the diagnosis.'))
        tried_and_failed = int(st.checkbox('Tried and failed generic alternative.'))
        contraindication = int(st.checkbox('Contraindication present for the drug.'))
        user_pa_X = pd.DataFrame({'bin': [payer], 
                                  'drug': [drug], 
                                  'correct_diagnosis': [correct_diagnosis],
                                  'tried_and_failed': [tried_and_failed],
                                  'contraindication': [contraindication]})

        pa_pipe = load(r"./models/saved-models/decision-tree-pa-approval.joblib")
        pa_pred = pa_pipe.predict(user_pa_X)[0]
        pa_prob = pa_pipe.predict_proba(user_pa_X)[0, 1]
        st.write("The prior authorization is **" + chance[pa_pred] + "** (" + str(int(100 * pa_prob)) + "%) to be approved.")

pages = {'Introduction': render_introduction_page,
         'Data': render_data_page,
         'Model': render_model_page,
         'Prototype': render_prototype_page}

st.title("Willow: A Medicinal Tree")
state = render_sidebar(list(pages.keys()))
pages[state]()
