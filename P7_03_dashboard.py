import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import urllib
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from lime.lime_tabular import LimeTabularExplainer
from PIL import Image
import shap

# warning on pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# paths and such
MODEL_FILE = 'model_file.sav'
FINAL_FILE = 'complete.pkl'
DESC_FILE = 'descriptions.pkl'
SHAP_EXP = 'shap_exp.sav'
SHAP_VAL = 'shap_val.pkl'
SUMMARY_SHAP = 'summary_shap.png'
GITHUB_ROOT = ('https://raw.githubusercontent.com/pipohipo/p7_dashboard/main/')

h_line = '''
---
'''

def load_obj(file: str):
    github_url = GITHUB_ROOT + file
    with urllib.request.urlopen(github_url) as open_file:
        return pickle.load(open_file)

# def load_obj(file: str):
#     return pickle.load(open(file, 'rb'))

@st.cache(suppress_st_warning=True)
def full_init():
    def initilize_desc():
        # list of features from descripton file
        df = load_obj(DESC_FILE)
        dflist = df['Feature'].tolist()
        return df, dflist

    desc, field_list = initilize_desc()

    def initialize_inputs():
        df = load_obj(FINAL_FILE)
        inputs_df = df.drop(columns=['TARGET', 'RISK_PROBA'])
        id_list = df.index.tolist()
        return df, inputs_df, id_list

    final, inputs, sk_id_list = initialize_inputs()

    def initialize_model():
        return make_pipeline(load_obj(MODEL_FILE)) #Load and create pipeline

    pipe = initialize_model()

    def initialize_shap():
        shap_exp = load_obj(SHAP_EXP)
        shap_val = load_obj(SHAP_VAL)
        return shap_exp, shap_val

    shap_explainer, shap_values = initialize_shap()

    return desc, field_list, final, inputs, sk_id_list, pipe, shap_explainer, shap_values

desc, field_list, final, inputs, sk_id_list, pipe, shap_explainer, shap_values = full_init()

# Apply threshold to positive probatilities
@st.cache
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

# Get predictions from FINAL and store the results
@st.cache(allow_output_mutation=True)
def get_og_predictions(final):
    # get labels and probabilities
    risk_flag = final['TARGET']
    risk_proba = final['RISK_PROBA']
    # return failure ratios
    pred_good = (risk_flag == 0).sum()
    pred_fail = (risk_flag == 1).sum()
    failure_ratio = round(pred_fail / (pred_good + pred_fail), 2)
    # result df
    results = final.copy()
    return results, failure_ratio, risk_proba

# Create original results
results, failure_ratio, risk_proba = get_og_predictions(final)

# What it says...
features_to_show = []

# Update predictions
@st.cache(allow_output_mutation=True)
def update_prediction(final, threshold):
    # og predictions
    risk_proba = final['RISK_PROBA']
    # new predictions
    risk_flag = to_labels(risk_proba, threshold)
    # return failure ratio
    pred_good = (risk_flag == 0).sum()
    pred_fail = (risk_flag == 1).sum()
    failure_ratio = round(pred_fail / (pred_good + pred_fail), 2)
    # update risk flags in results
    results['TARGET'] = risk_flag
    return results, failure_ratio

st.write(
"""
# Credit Scoring
"""
)

###############################################################
# COOL SIDEBAR
###############################################################
st.sidebar.header('Inputs')
st.sidebar.markdown(h_line)
st.markdown(h_line)
# Failure ratio controler
st.sidebar.subheader('Failure Ratio Control')
st.sidebar.write('Failure Ratio', failure_ratio)

def threshold_prediction():
    # Threshold slider
    new_threshold = st.sidebar.slider(label='Threshold:', min_value=0., value=0.5, max_value=1.)
    # Update results
    new_failure_ratio = failure_ratio
    results, new_failure_ratio = update_prediction(final, new_threshold)
    # Write on sidebar
    st.sidebar.write('Selected Failure Ratio', new_failure_ratio)
    return new_threshold

current_threshold = threshold_prediction()

st.sidebar.markdown(h_line)

# Client selection (sk_id)
st.sidebar.subheader('Client Selection')
def select_client():
    sk_id_select = st.sidebar.selectbox('Client ID', sk_id_list, 0)
    sk_row = results.loc[[sk_id_select]]
    return sk_row, sk_id_select

selected_sk_row, selected_sk_id = select_client()

st.sidebar.markdown(h_line)

# Features description
st.sidebar.subheader('Feature description')
def feat_description():
    select_feat = st.sidebar.selectbox('Select a feature', field_list, 0)
    select_desc = desc[desc['Feature'] == select_feat]['Description']
    pd.options.display.max_colwidth = len(select_desc)
    return select_desc

txt_feat_desc = feat_description()

st.sidebar.text(txt_feat_desc)

###############################################################
# MAIN PAGE
###############################################################
st.subheader('Selected Client')
st.write()
st.write(selected_sk_row)

st.markdown(h_line)

st.subheader('Applications sample demo')

def application_samples_component():
    if st.button('Generate sample'):
        st.markdown('predicted __without__ difficulty to repay - sample')
        st.write(results[results['TARGET'] == 0].sample(3))
        st.markdown('predicted __with__ difficulty to repay - sample')
        st.write(results[results['TARGET'] == 1].sample(3))

application_samples_component()

st.markdown(h_line)

# LIME: Local Interpretable Model-agnostic Explanations
st.subheader('LIME explainer')

def lime_explaination(inputs, results, selected_sk_id):
    # Write slider settings
    st.write('Set the number of features to analyse')
    nb_features = st.slider(label='Features', min_value=1, value=10, max_value=inputs.shape[1])
    st.write('Set the number of similar clients to compare with (based on feature importance)')
    nb_neighbors = st.slider(label='Similar clients', min_value=2, value=5, max_value=30)

    # If button is activated... then do the thingy
    if st.button('Generate LIME'):
        with st.spinner('Calculating...'): # doing the thingy...
            # create explainer
            lime_explainer = LimeTabularExplainer(
                training_data=inputs.values, 
                mode='classification', 
                training_labels=results['TARGET'], 
                feature_names=inputs.columns)
            # lime explanation for a given SK_ID_CURR value (application/client)
            exp = lime_explainer.explain_instance(
                inputs.loc[selected_sk_id].values, 
                pipe.predict_proba, 
                num_features=nb_features)
            
            # blahblah details
            st.write('LIME for the selected client:')
            st.write('Positive values in __Red__ means __Support__ the Class 1: Failure Risk')
            st.write('Negative values in __Green__ means __Contradict__ the Class 1: Failure Risk')
            
            # get features_to_show list
            id_cols = [item[0] for item in exp.as_map()[1]]
            # inputs restricted to the features_to_show
            df_lime = inputs.filter(inputs.columns[id_cols].tolist())
            
            # compute inputs for plots
            exp_list = exp.as_list()
            vals = [x[1] for x in exp_list]
            names = [x[0] for x in exp_list]
            
            axisgb_colors = ['#FABEC0' if x > 0 else '#B4F8C8' for x in vals]
            
            vals.reverse()
            names.reverse()
            
            colors = ['red' if x > 0 else 'green' for x in vals]
            
            pos = np.arange(len(exp_list))
            
            # create plot
            plt_tab = plt.figure()
            plt.barh(pos, vals, align='center', color=colors)
            plt.yticks(pos, names)
            plt.title('Local explanation for Class 1: Failure Risk')
            st.pyplot(plt_tab)
            
            # find nb_neighbors nearest neighbors
            nearest_neighbors = NearestNeighbors(n_neighbors=nb_neighbors, radius=0.3)
            nearest_neighbors.fit(df_lime)
            
            neighbors = nearest_neighbors.kneighbors(
                X=df_lime.loc[[selected_sk_id]], #current observation
                n_neighbors=nb_neighbors+1, #it gives the X value in the 0 position so we need one more
                return_distance=False)[0]
            neighbors = np.delete(neighbors, 0)
            
            # compute values for neighbors
            df_lime['TARGET'] = results['TARGET']
            neighbors_values = pd.DataFrame(
                df_lime.iloc[neighbors].median(), 
                index=df_lime.columns, 
                columns=['Neighbors_Median'])
            
            st.write('__- Neighbors Risk Flag averaged__', neighbors_values.Neighbors_Median.tail(1).values[0])

            client_values = df_lime.loc[[selected_sk_id]].T
            client_values.columns = ['Client']

            class1_values = pd.DataFrame(
                df_lime[df_lime['TARGET'] == 1].median(), 
                index=df_lime.columns, 
                columns=['Class_1_Median'])

            class0_values = pd.DataFrame(
                df_lime[df_lime['TARGET'] == 0].median(), 
                index=df_lime.columns, 
                columns=['Class_0_Median'])
                
            classes_values = pd.concat([class0_values, class1_values, neighbors_values, client_values], axis=1)

            fig, axs = plt.subplots(nb_features, sharey='row', figsize=(10, 5*nb_features))
            colorsList = ('#FB475E', '#019992', '#44EE77', '#FFB001')
            
            for i in np.arange(0, nb_features):
                axs[i].barh(classes_values.T.index, classes_values.T.iloc[:, i], color=colorsList)
                axs[i].set_title(str(classes_values.index[i]), fontweight="bold")
                axs[i].patch.set_facecolor(axisgb_colors[i])
            st.write('Client comparison with its neighbors, Class 0 and Class 1 medians')
            st.write('__Lightred__ (__Lightgreen__) background means __Support__ (__Contradict__) for the Class 1: Failure Risk')
            st.pyplot(fig)

lime_explaination(inputs, results, selected_sk_id)

# SHAP
st.subheader('SHAP explainer')

def shap_explaination(selected_sk_id):
    if st.button('Generate SHAP'):
        with st.spinner('Calculating...'):
            st.write('__SH__apley __A__dditive ex__P__lanations: how the most important features impact on class prediction')

            idx = inputs.index.get_loc(selected_sk_id)

            client_fig = shap.force_plot(
                shap_explainer.expected_value,
                shap_values[idx, :],
                inputs.iloc[idx, :])
            client_fig_html = f"<head>{shap.getjs()}</head><body>{client_fig.html()}</body>"
            components.html(client_fig_html, height=150)

            feat_fig = shap.force_plot(
                shap_explainer.expected_value,
                shap_values[:50, :],
                inputs.iloc[:50, :])
            feat_fig_html = f"<head>{shap.getjs()}</head><body>{feat_fig.html()}</body>"
            components.html(feat_fig_html, height=350)

            # summ_fig = shap.summary_plot(shap_values, inputs)
            # summ_fig_html = f"<head>{shap.getjs()}</head><body>{summ_fig.html()}</body>"
            # components.html(summ_fig_html, heigth=350)

shap_explaination(selected_sk_id)

st.image(Image.open(SUMMARY_SHAP), caption='Shap summary plot')