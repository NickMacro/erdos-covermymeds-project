import streamlit as st
import pandas as pd
from bokeh.plotting import figure
from joblib import load

# global variable used to provide explanation of rational for each reject_code
CODE_TRANSLATION = {70: "the drug is not on the formulary and not covered by the plan",
                    75: "the drug is on the formulary but requires prior authorization",
                    76: "the plan limitations have been exceeded"}


@st.cache
def load_data(claims_path, pa_path, bridge_path):
    """ Function to load the data required for data exploration and populating the user input fields.
    The function is cached (memoized) using the @st.cache decorator to improve performance.
    Args:
        claims_path (str): Path to the claims csv file.
        pa_path (str): Path to the prior authorization csv file.
        bridge_path (str): Path to the bridge csv file.

    Returns:
        claims_df (DataFrame): Pandas DataFrame containing the claims data.
        pa_combined_df (DataFrame): Pandas Dataframe containing the merged claims and prior authorization data.
    """

    # claims that are approved missing value in the reject_code
    # the missing value is filled with 0 and the column is typecast to int
    # this is to preserve consistency with the data used in the pipeline
    # and prevents problems with the encoding step of the pipeline
    claims_df = pd.read_csv(claims_path).fillna(0)
    claims_df.loc[:, 'reject_code'] = claims_df['reject_code'].astype(int)

    # the pa and bridge dataframes do not need special treatment
    pa_df = pd.read_csv(pa_path)
    bridge_df = pd.read_csv(bridge_path)

    # the prior authorization data is a merge of the three above dataframes
    # this will remove authorized claims because they do not have a valid dim_pa_id
    # ids are dropped because they are not used once the data is merged
    pa_combined_df = bridge_df.merge(claims_df, on='dim_claim_id').merge(pa_df, on='dim_pa_id').drop(columns=['dim_claim_id', 'dim_pa_id'])
    return claims_df, pa_combined_df


def render_sidebar(page_names):
    """ Function used to render the sidebar text.
    Args:
        page_names (list): A list of strings with the names of each page that can be rendered.

    Returns:
        state (str): The state selected using the radio button.
    """

    # user selects the state using the radio element which will
    # determine the page that is rendered
    state = st.sidebar.radio('Table of Contents', page_names)
    st.sidebar.write("This project was performed as part of [The Erdős Institute](https://www.erdosinstitute.org/)'s Spring 2021 Data Science Bootcamp.")
    st.sidebar.write("The data were provided by [CoverMyMeds](https://www.covermymeds.com/main/).")
    st.sidebar.write("Created by [Nick Macro](https://www.linkedin.com/in/nickmacro/), [Sandrine Müller](https://www.linkedin.com/in/sandrinermuller/), and [Tomas Kasza](https://www.linkedin.com/in/tomas-kasza/).")
    return state


def render_introduction_page():
    """ Function used to render the introduction page.
    This function is used to contain the content of the intoduction page. The content is placed in a function
    in order to simplify the rendering process and keep the content of each page organized.
    """

    st.header("Introduction")
    st.write("Patients can encounter difficulties with insurance coverage of specific drugs, depending on the insurance payer. We are hoping to solve the following problems:")
    st.write("""
    1. Can a claim’s rejection be predicted using claim data? If so, what factors tend to influence the rejection of a claim?
    2. Can the reason for rejection be predicted using the claim data?
    3. Can the approval of a PA be predicted using claim and PA data? If so, what factors tend to influence the approval of a PA? 
    """)


def render_data_page():
    """ Function used to render the data page.
    This function is used to contain the content of the data page. The content is placed in a function
    in order to simplify the rendering process and keep the content of each page organized.
    """

    st.header("Data")
    st.write("The data consists of four tables of simulated data pertaining to patients’ access to drugs and prior authorizations (PA) due to rejection by the payer.")
    
    st.write("There are ~1.3 million claims across three years from January 1, 2017 to December 31, 2019. The simulated pharmacy claim-level data provided is for three drugs (A, B, C) and four payers (417380, 417614, 417740, 999001). ~556k claims are rejected and require a prior authorization (PA). There are three rejection codes provided:")
    st.write("""
- "70" for a drug that is not covered by the plan and not on formulary
- "75" for a drug on the formulary that does not have preferred status and requires a PA
- "76" for a drug that is covered but the plan limitations have been exceeded.
""")
    st.write("The PA data contains four binary categories indicating whether the drug was prescribed for the correct diagnosis (80% of PAs), has tried and failed a generic alternative (50% of PAs), if the patient has an associated contraindication (20% of PAs), and whether the PA was approved (73% of PAs).")
    claims_df, pa_combined_df = load_data("./data/processed/dim_claims_train.csv",
                                         "./data/processed/dim_pa_train.csv",
                                         "./data/processed/bridge_train.csv")

    # sorted list of each of the payers for future use
    payers = sorted(claims_df['bin'].unique())
    payer = int(st.selectbox('Enter payer ID.', payers))

    # user payer selection is used to segment the claims data by payer
    claim_view = claims_df.loc[claims_df['bin'] == payer]
    # empty DataFrame used to construct the table output
    drug_coverage = pd.DataFrame()
    # count is used to determine the number of rows in each group
    drug_coverage.loc[:, 'Claims Submitted'] = claim_view.groupby('drug')['pharmacy_claim_approved'].count()
    # mean is used to determine the percentage of claims approved because an approval is 1 and denial 0.
    # the mean therefore provides (# approvals) / (# claims)
    drug_coverage.loc[:, 'Claim Approval %'] = round(100 * claim_view.groupby('drug')['pharmacy_claim_approved'].mean())
    # a column for drug with a heading is desired
    # the index is not provided a heading in a streamlit table
    # a column is created using the index and the index is set to empty strings
    drug_coverage.loc[:, 'Drug'] = drug_coverage.index
    drug_coverage.index = ['' for _ in range(len(drug_coverage))]
    # adjust the order of appearance of each column
    drug_coverage = drug_coverage[['Drug', 'Claims Submitted', 'Claim Approval %']]
    st.table(drug_coverage)

    # segment the pa data by payer
    pa_view = pa_combined_df.loc[pa_combined_df['bin'] == payer]

    # sorted list of each drug for future use
    drugs = sorted(pa_view['drug'].unique())

    # write a statement about the approval of each drug for the selected payer
    for drug in drugs:
        # in the current data, each payer-drug combination has a single reject_code
        # the code below will need to be updated
        # if the data changes to have multiple reject_codes for each combination

        # take the first reject_code, because there is only a single reject_code for a drug-payer combination
        reject_code = pa_view.loc[pa_view['drug'] == drug]['reject_code'].iloc[0]
        # rejection_rate = 1 - approval_rate
        # approval rate can be determined using the mean (as described above)
        drug_rejection = round(100 * (1 - claim_view.loc[claim_view['drug'] == drug]['pharmacy_claim_approved'].mean()))
        text = f"If rejected ({drug_rejection}%), drug {drug} is always rejected because {CODE_TRANSLATION[reject_code]} (code {reject_code})."
        st.write(text)

    drug = st.selectbox('Enter drug name.', drugs)

    # dictionary with the pretty display name of each PA feature name
    pa_names = {'contraindication': 'Contraindication',
                'tried_and_failed': 'Failed Generic',
                'correct_diagnosis': 'Correct Diagnosis'}

    # segment the PA data to a single payer-drug combination
    pa_view = pa_combined_df.loc[(pa_combined_df['bin'] == payer) & (pa_combined_df['drug'] == drug)]
    # empty DataFrame used to calculate the approval rate for each combination of features
    rejection_table = pd.DataFrame()
    # count and mean usage are described above
    # the keys of pa_names are the feature names of each feature in the PA data
    rejection_table.loc[:, 'pa_ammount'] = pa_view.groupby(list(pa_names.keys()))['pa_approved'].count()
    rejection_table.loc[:, 'pa_rate'] = pa_view.groupby(list(pa_names.keys()))['pa_approved'].mean()
    rejection_table = rejection_table.sort_values('pa_rate', ascending=False)

    # empty DataFrame used to construct the table output
    rejection_display = pd.DataFrame()
    for key, value in pa_names.items():
        # add a column to the output table for each feature in the PA data
        # create columns for each index in the multilevel indexed table generated above
        # typecast the outputs into bools to make the output easier to read
        rejection_display.loc[:, value] = rejection_table.index.get_level_values(key).astype(bool)
    # make the output columns pretty and remove the index
    rejection_display.loc[:, 'PAs Submitted'] = rejection_table['pa_ammount'].values
    rejection_display.loc[:, 'PA Approval %'] = 100 * rejection_table['pa_rate'].values
    rejection_display.loc[:, 'PA Approval %'] = rejection_display['PA Approval %'].round()
    rejection_display.index = ['' for _ in range(len(rejection_display))]
    st.table(rejection_display)

    # write statements about how each PA feature changes the outcome for
    # the drug-payer combination selected above
    for name in pa_names.keys():
        # the approval rate if the feature is true
        true_rate = pa_view.loc[pa_view[name] == 1]['pa_approved'].mean()
        # the approval rate if the feature is false
        false_rate = pa_view.loc[pa_view[name] == 0]['pa_approved'].mean()

        rate_delta = int(round(100 * (true_rate - false_rate), 0))
        if rate_delta >= 0:
            change = 'increases'
        else:
            change = 'decreases'
        st.write(f"{pa_names[name]} {change} the PA Approval % by {abs(rate_delta)}%.")


def bokeh_image_viewer(left, right, top, bottom, img_url):
    """ Function used to render the interactive image viewers using Bokeh.
    Args:
        left (float): Left edge of the viewing box.
        right (float): Right edge of the viewing box.
        top (float): Top edge of the viewing box.
        bottom (float): Bottom edge of the viewing box.
        img_url (str): URL of the image displayed by the viewer.

    Returns:
        f (bokeh figure): Bokeh figure 
    """

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
    """ Function used to render the model page.
    This function is used to contain the content of the model page. The content is placed in a function
    in order to simplify the rendering process and keep the content of each page organized.
    """

    # urls of each decision tree image
    # the version of bokeh (2.2.*) that works with the current version of streamlit (0.82.*)
    # does not work with providing the relative path of the image file
    # urls to the image files are provided instead
    img_urls = {'claim_approval': 'https://github.com/NickMacro/erdos-covermymeds-project/raw/main/models/saved-model-figures/claim-approval-tree.png',
                'reject_code': 'https://github.com/NickMacro/erdos-covermymeds-project/raw/main/models/saved-model-figures/reject-code-tree.png',
                'pa_approval': 'https://github.com/NickMacro/erdos-covermymeds-project/raw/main/models/saved-model-figures/pa-approval-tree.png'}

    st.header("Model")
    st.write("A decision tree model was found to be the best performing model for both:")
    st.write("""1. Predicting if a claim will be approved by the payer.
2. Predicting the reject code for rejected claims.
3. Predicting if a prior authorization will be approved by the payer.""")

    # returns false by default
    interactive_plots = st.checkbox('Trouble viewing trees? Use an interactive viewer.')

    # the three codeblocks have the same logic for each plot
    # interactive_plots determines the method of rendering the image
    # the viewer position (left, right, bottom, top) 
    # needs to be placed in different locations based
    # on the size of the image rendered
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
    """ Function used to render the prototype page.
    This function is used to contain the content of the prototype page. The content is placed in a function
    in order to simplify the rendering process and keep the content of each page organized.
    """

    st.header("Prototype")
    st.write("The decision tree models can be used to predict the approval of future claims and prior authorizations.")

    # the dataframe is only used to provide a list of the payers (bin)
    # and drugs (drug)
    claims_df, _ = load_data("./data/processed/dim_claims_train.csv",
                                          "./data/processed/dim_pa_train.csv",
                                          "./data/processed/bridge_train.csv")

    payers = sorted(claims_df['bin'].unique())
    drugs = sorted(claims_df['drug'].unique())
    payer = int(st.selectbox('Enter payer ID.', payers))
    drug = st.selectbox('Enter drug name.', drugs)

    # construct a DataFrame input for prediction by the claim model
    user_claims_X = pd.DataFrame({'bin': [payer], 'drug': [drug]})

    # load the model and use it to predict the approval of the claim specified by the user input
    claims_pipe = load(r"./models/saved-models/decision-tree-claim-approval.joblib")
    # obtain the first (and only) element of the output
    claim_pred = claims_pipe.predict(user_claims_X)[0]
    # obtain the first (and only) element of the output and second column (the likelyhood of approval)
    claim_prob = claims_pipe.predict_proba(user_claims_X)[0, 1]

    # the term (un)likely is used to reflect the predicted outcome
    chance = {0: 'unlikely', 1: 'likely'}

    st.write("The prescription is **" + chance[claim_pred] + "** (" + str(int(100 * claim_prob)) + "%) to be approved.")

    # only perform the below steps if rejection is predicted
    if not claim_pred:
        # load the model and use it to predict the reject_code of claim specified by the user input
        reject_pipe = load(r"./models/saved-models/decision-tree-reject-code.joblib")
        # the input is the same as for the claim model
        # obtain the first (and only) element of the output
        reject_pred = reject_pipe.predict(user_claims_X)[0]

        st.write(f'The prescription is likely to require a prior authorization because {CODE_TRANSLATION[reject_pred]}. Fill out the following details to determine if the prior authorization is likely to be accepted.')
        correct_diagnosis = int(st.checkbox('Drug is appropriate for the diagnosis.'))
        tried_and_failed = int(st.checkbox('Tried and failed generic alternative.'))
        contraindication = int(st.checkbox('Contraindication present for the drug.'))
        # construct a DataFrame input for prediction by the pa model
        user_pa_X = pd.DataFrame({'bin': [payer], 
                                  'drug': [drug], 
                                  'correct_diagnosis': [correct_diagnosis],
                                  'tried_and_failed': [tried_and_failed],
                                  'contraindication': [contraindication]})

        # load the model and use it to predict the approval of the pa specified by the user input
        pa_pipe = load(r"./models/saved-models/decision-tree-pa-approval.joblib")
        # obtain the first (and only) element of the output
        pa_pred = pa_pipe.predict(user_pa_X)[0]
        # obtain the first (and only) element of the output and second column (the likelyhood of approval)
        pa_prob = pa_pipe.predict_proba(user_pa_X)[0, 1]
        st.write("The prior authorization is **" + chance[pa_pred] + "** (" + str(int(100 * pa_prob)) + "%) to be approved.")

# create with each page name and the content
pages = {'Introduction': render_introduction_page,
         'Data': render_data_page,
         'Model': render_model_page,
         'Prototype': render_prototype_page}

# always display the title
st.title("Willow: A Medicinal Tree")
# render the sidebar with the page names from pages
# obtain the page to be generated
state = render_sidebar(list(pages.keys()))
# render the page specified
pages[state]()
