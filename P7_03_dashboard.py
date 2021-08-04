import streamlit as st #https://docs.streamlit.io/en/stable/api.html
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
FINAL_CATS_FILE = 'complete_cats.pkl'
DESC_FILE = 'descriptions.pkl'
# SHAP_EXP = 'shap_exp.sav'
# SHAP_VAL = 'shap_val.pkl'
SUMMARY_SHAP = 'summary_shap.png'
GITHUB_ROOT = ('https://raw.githubusercontent.com/pipohipo/p7_dashboard/main/')

h_line = '''
---
'''

###############################################################
# PRELOAD
###############################################################

def load_obj(file: str):
    github_url = GITHUB_ROOT + file
    with urllib.request.urlopen(github_url) as open_file:
        return pickle.load(open_file)

@st.cache(suppress_st_warning=True)
def full_init():
    def initilize_desc():
        # list of features from descripton file
        df = load_obj(DESC_FILE)
        dflist = df['features'].tolist()
        return df, dflist

    desc, field_list = initilize_desc()

    def initialize_inputs():
        df = load_obj(FINAL_FILE)
        inputs_df = df.drop(columns=['TARGET', 'RISK_PROBA'])
        id_list = df.index.tolist()
        return df, inputs_df, id_list

    final, inputs, sk_id_list = initialize_inputs()

    def initialize_inputs_cats():
        return load_obj(FINAL_CATS_FILE)

    final_cats = initialize_inputs_cats()        

    def initialize_model():
        return make_pipeline(load_obj(MODEL_FILE)) #Load and create pipeline

    pipe = initialize_model()

    # def initialize_shap():
    #     shap_exp = load_obj(SHAP_EXP)
    #     shap_val = load_obj(SHAP_VAL)
    #     return shap_exp, shap_val

    # shap_explainer, shap_values = initialize_shap()

    return desc, field_list, final, inputs, sk_id_list, final_cats, pipe

desc, field_list, final, inputs, sk_id_list, final_cats, pipe = full_init()

# What it says...
features_to_show = []

st.write(
"""
# Credit Scoring
"""
)

###############################################################
# COOL SIDEBAR
###############################################################
# Client selection (sk_id)
st.sidebar.header('Client Selection')
def select_client():
    sk_id_select = st.sidebar.selectbox('SK_ID_CURR', sk_id_list, 0) #117082 Class 1 client
    sk_row = final.loc[[sk_id_select]]
    return sk_row, sk_id_select

selected_sk_row, selected_sk_id = select_client()

st.sidebar.markdown(h_line)

st.sidebar.image(Image.open('pred_distrib.png'), caption='Target distribution')

st.sidebar.markdown(h_line)

# Features description
st.sidebar.header('Feature description')
def feat_description():
    select_feat = st.sidebar.selectbox('Select a feature', field_list, 0)
    select_desc = desc[desc['features'] == select_feat]['definitions'].values[0]
    pd.options.display.max_colwidth = len(select_desc)
    return select_desc

txt_feat_desc = feat_description()

st.sidebar.text(txt_feat_desc)

###############################################################
# MAIN PAGE
###############################################################
st.write('Default risk probability:', selected_sk_row['RISK_PROBA'].values[0])
st.write('Category:', selected_sk_row['TARGET'].values[0])
st.write('Client info:', selected_sk_row.drop(columns=['RISK_PROBA', 'TARGET']))

st.markdown(h_line)

st.subheader('Random applications sample')

def application_samples_component():
    st.write('Sample size:')
    nb_clients_sample = st.number_input(
        label='Number of clients', 
        min_value=1,
        max_value=final.shape[0],
        format='%i')
    if st.button('Generate sample'):
        st.write(final.sample(nb_clients_sample))

application_samples_component()

st.markdown(h_line)

# LIME: Local Interpretable Model-agnostic Explanations
st.subheader('LIME explainer')

def lime_explaination(inputs, final, selected_sk_id):
    # Write slider settings
    st.write('Client vs similar profiles')
    nb_features = st.slider(
        label='Set the number of features to analyse', 
        min_value=1, 
        value=10, 
        max_value=inputs.shape[1])
    nb_neighbors = st.slider(
        label='Set the number of similar profiles to compare with', 
        min_value=0, 
        value=4, 
        max_value=10)

    # If button is activated... then do the thingy
    if st.button('Generate LIME'):
        with st.spinner('Calculating...'): # doing the thingy...
            # create explainer
            lime_explainer = LimeTabularExplainer(
                training_data=inputs.values, 
                mode='classification', 
                training_labels=final['TARGET'], 
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
            
            if nb_neighbors > 0:
                # find nb_neighbors nearest neighbors
                nearest_neighbors = NearestNeighbors(n_neighbors=nb_neighbors, radius=0.3)
                nearest_neighbors.fit(df_lime)
            
                neighbors = nearest_neighbors.kneighbors(
                    X=df_lime.loc[[selected_sk_id]], #current observation
                    n_neighbors=nb_neighbors+1, #it gives the X value in the 0 position so we need one more
                    return_distance=False)[0]
                neighbors = np.delete(neighbors, 0)
            
                # compute values for neighbors
                df_lime['TARGET'] = final['TARGET']
                
                neighbors_values_int = df_lime.iloc[neighbors].select_dtypes(include=['int8']).mean().round(0)
                neighbors_values_float = df_lime.iloc[neighbors].select_dtypes(include=['float16', 'float32']).mean()
                neighbors_values = pd.concat([neighbors_values_int, neighbors_values_float]).reindex(df_lime.columns.tolist())

                neighbors_values = pd.DataFrame(
                    neighbors_values,
                    index=df_lime.columns, 
                    columns=['neighbors_mean'])

                st.write('__- Neighbors risk average__', neighbors_values.neighbors_mean.tail(1).values[0])

            else:
                # compute values for neighbors
                df_lime['TARGET'] = final['TARGET']

                neighbors_values = pd.DataFrame(
                    0,
                    index=df_lime.columns, 
                    columns=['neighbors_mean'])

            client_values = df_lime.loc[[selected_sk_id]].T
            client_values.columns = ['client']

            class_1_values_int = df_lime[df_lime['TARGET'] == 1].select_dtypes(include=['int8']).mean().round(0)
            class_1_values_float = df_lime[df_lime['TARGET'] == 1].select_dtypes(include=['float16', 'float32']).mean()
            class_1_values = pd.concat([class_1_values_int, class_1_values_float]).reindex(df_lime.columns.tolist())
            class_1_values = pd.DataFrame(
                class_1_values, 
                index=df_lime.columns,
                columns=['class_1_mean'])

            class_0_values_int = df_lime[df_lime['TARGET'] == 0].select_dtypes(include=['int8']).mean().round(0)
            class_0_values_float = df_lime[df_lime['TARGET'] == 0].select_dtypes(include=['float16', 'float32']).mean()
            class_0_values = pd.concat([class_0_values_int, class_0_values_float]).reindex(df_lime.columns.tolist())
            class_0_values = pd.DataFrame(
                class_0_values, 
                index=df_lime.columns,
                columns=['class_0_mean'])

            classes_values = pd.concat([client_values, neighbors_values, class_0_values, class_1_values], axis=1)
            classes_values.replace([-np.inf, np.inf], 1, inplace=True)
            classes_values.fillna(0, inplace=True)

            fig, axs = plt.subplots(nb_features, sharey='row', figsize=(10, 5*nb_features))
            colorsList = ('#FB475E', '#019992', '#44EE77', '#FFB001')
            
            for i in np.arange(0, nb_features):
                axs[i].barh(classes_values.T.index, classes_values.T.iloc[:, i], color=colorsList)
                axs[i].set_title(str(classes_values.index[i]), fontweight="bold")
                axs[i].patch.set_facecolor(axisgb_colors[i])
            st.write('Client comparison with its neighbors, Class 0 and Class 1 means')
            st.write('__Lightred__ (__Lightgreen__) background means __Support__ (__Contradict__) for the Class 1: Failure Risk')
            st.pyplot(fig)

lime_explaination(inputs, final, selected_sk_id)

# SHAP
st.subheader('SHAP Summary plot')

# SHAP for each client is disable since it takes a lot of time to calculate and the SHAP model is over a 100MB
# def shap_explaination(selected_sk_id):
#     if st.button('Generate SHAP'):
#         with st.spinner('Calculating...'):
#             st.write('__SH__apley __A__dditive ex__P__lanations: how the most important features impact on class prediction')

#             idx = inputs.index.get_loc(selected_sk_id)

#             client_fig = shap.force_plot(
#                 shap_explainer.expected_value,
#                 shap_values[idx, :],
#                 inputs.iloc[idx, :])
#             client_fig_html = f"<head>{shap.getjs()}</head><body>{client_fig.html()}</body>"
#             components.html(client_fig_html, height=150)

#             feat_fig = shap.force_plot(
#                 shap_explainer.expected_value,
#                 shap_values[:50, :],
#                 inputs.iloc[:50, :])
#             feat_fig_html = f"<head>{shap.getjs()}</head><body>{feat_fig.html()}</body>"
#             components.html(feat_fig_html, height=350)

# shap_explaination(selected_sk_id)

st.image(Image.open(SUMMARY_SHAP), caption='SHAP summary plot')