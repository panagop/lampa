import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystrata
import pyexcel
import json
import pickle
from bokeh.plotting import figure
from bokeh.themes import built_in_themes
from bokeh.io import curdoc

import pystrata

from lampa.input import LPyStrataInput
from lampa.input import LTimeSeriesMotion
from lampa.input import LSoilType, LDarendeliSoilType
from lampa.input import LLayer
from lampa.project import LProject

from io import StringIO



st.title('1-D Seismic Site Response Analysis')

st.text('This is a web app for one dimensional ground response analysis \nusing data from seismic motions for the development of \nAcceleration Response Spectra and Spectral Ratio \nPystrata library etc etc')

# load ini json file
lpsi = LPyStrataInput.from_json_file('streamlit/ini.json')
# with open('ini.pickle', "rb") as pfile:
#     lpsi = pickle.load(pfile)

lproject = LProject(lpsi)


############################################################################################################
# Sidebar - Files - Input parameters - Options
############################################################################################################

###################
# File
###################

# st.sidebar.markdown('## Project input file')
with st.sidebar.expander(label='Project input file', expanded=False):
    st.download_button('Save input file', LPyStrataInput.schema().dumps(
        lpsi), 'lampa.json', 'application/json')

    st.write('---')

    uploaded_input = st.file_uploader('Load input file', type=['json'])
    if uploaded_input is not None:
        file_details = {"FileName": uploaded_input.name,
                        "FileType": uploaded_input.type, "FileSize": uploaded_input.size}
        st.write(file_details)

        xxx = uploaded_input.read()
        st.write(type(xxx))
        lpsi = LPyStrataInput.from_json_binary(uploaded_input.read())
        # st.write(str(xxx))

        # stringio = StringIO(uploaded_input.getvalue().decode("utf-8"))
        # lpsi = LPyStrataInput.from_dict(stringio.read())
        # st.write((stringio.read()))
        # lpsi = LPyStrataInput.from_json_file(stringio.read())
        # lproject = LProject(lpsi)
        # with open(uploaded_input, "rb") as pfile:
        #     lpsi = pickle.load(pfile)

st.sidebar.markdown('---')

###################
# Seismic Motion
###################
# st.sidebar.markdown('## Seismic Motion - Input')
st_seismic_motion_sidebar = st.sidebar.expander(
    label='Load input seismic motion from file', expanded=False)
with st_seismic_motion_sidebar:
    accel_filetype = st.radio('Select filetype', options=[
                              'xlsx', 'txt', 'csv'])  # , 'AT2'
    uploaded_accel = st.file_uploader(
        'Upload your file here', type=[accel_filetype])

    if uploaded_accel is not None:
        if accel_filetype == 'txt':
            ltsm = LTimeSeriesMotion.from_txt(uploaded_accel)
        elif accel_filetype == 'csv':
            ltsm = LTimeSeriesMotion.from_csv(uploaded_accel)
        elif accel_filetype == 'xlsx':
            ltsm = LTimeSeriesMotion.from_excel(uploaded_accel)

        lpsi.time_series_motion = ltsm

st.sidebar.markdown('---')


###################
# Soil Layers
###################

# st.sidebar.markdown('## Soil Layers')
with st.sidebar.expander(label='Soil Layers', expanded=False):

    layers = lpsi.layers

    no_layers = st.slider('Number of Layers', 1, 20, len(layers))

    layer_names = [f'Layer {i}' for i in range(1, len(layers)+1)]

    layer_tabs = st.tabs(layer_names)

    for i, layer in enumerate(layers):
        with layer_tabs[i]:
            st_layer = {}


#     layer_names = [f'Layer {i}' for i in range(1, no_layers+1)]

#     layers = []

#     layer_tabs = st.tabs(layer_names)

#     for i in range(0, no_layers):
#         with layer_tabs[i]:
#             layer = {}
#             layer_tabs[i].header(f'Layer {i+1} properties')
#             layer['unit_wt'] = layer_tabs[i].number_input(
#                 key=f'unit_wt {i+1}', label='unit weight of the material [kN/m³]', value=18.0)
#             layer['plas_index'] = layer_tabs[i].number_input(
#                 key=f'plas_index {i+1}', label='plasticity index [percent]', value=0.0)
#             layer['ocr'] = layer_tabs[i].number_input(
#                 key=f'ocr {i+1}', label='over-consolidation ratio', value=1)
#             layer['stress_mean'] = layer_tabs[i].number_input(
#                 key=f'stress_mean {i+1}', label='mean effective stress [kN/m²]', value=200)
#             layer['freq'] = layer_tabs[i].number_input(
#                 key=f'freq {i+1}', label='excitation frequency [Hz]', value=30)
#             layer['num_cycles'] = layer_tabs[i].number_input(
#                 key=f'num_cycles {i+1}', label='number of cycles of loading', value=500)
#             # layer_tabs[i].number_input(key=f'strains {i+1}', label = 'shear strains levels [decima', value=18.0)
#             layers.append(layer)

# # st.write(layers)

# # selected_layer = st.select_slider('select layer', options=range(1, no_layers+1))
# # st.write(f'selected_layer: {selected_layer}')
# # st.write(f'unit_wt for selected_layer: {layers[selected_layer-1]["unit_wt"]}')


# list_layers = []

# list_layers.append(pystrata.site.Layer(
#     pystrata.site.DarendeliSoilType(16.0, plas_index=0, ocr=1, stress_mean=40),
#     5,
#     140,
# ))

# list_layers.append(pystrata.site.Layer(
#     pystrata.site.DarendeliSoilType(
#         18.0, plas_index=25, ocr=1, stress_mean=215),
#     15,
#     250,
# ))

# list_layers.append(pystrata.site.Layer(
#     pystrata.site.DarendeliSoilType(
#         20.0, plas_index=0, ocr=1, stress_mean=650),
#     30,
#     450,
# ))

# list_layers.append(pystrata.site.Layer(
#     pystrata.site.SoilType("Rock", 23.0, None, 0.02), 0, 1200))

st.sidebar.markdown('---')

############################################################################################################
# Results
############################################################################################################
freqs = np.logspace(-0.5, 2, num=500)

# Sidebar

# st.sidebar.markdown('## Results')
st_results_sidebar_expander = st.sidebar.expander(
    label='Results options', expanded=False)
with st_results_sidebar_expander:
    results_damping = st.number_input(
        label='damping', value=0.05, format='%.3f')


st.sidebar.markdown('---')

###################
# General Options
###################

# st.sidebar.markdown('## General Options')
st_options_sidebar_expander = st.sidebar.expander(
    label='General Options', expanded=False)
with st_options_sidebar_expander:
    inteactive_charts = st.checkbox(label='Interactive charts', value=False)

    interactive_chart_theme = st.selectbox(label='Interactive chart theme',
     options=['caliber', 'dark_minimal', 'light_minimal', 'night_sky', 'contrast'])

    # https://discuss.streamlit.io/t/bokeh-theming/15302
    doc=curdoc()
    doc.theme = interactive_chart_theme





############################################################################################################
# Main view
############################################################################################################

###################
# Seismic Motion
###################


st.markdown('## Seismic motion')

st_seismic_motion_main_expander = st.expander(
    label='Seismic motion', expanded=False)
with st_seismic_motion_main_expander:

    st.markdown('### Accelerations')
    if inteactive_charts:
        dt_accel = pd.DataFrame(data={'Time': lpsi.time_series_motion.to_pystrata.times,
                                'Acceleration (g)': lpsi.time_series_motion.to_pystrata.accels})
        st.line_chart(data=dt_accel, x='Time', y='Acceleration (g)')
    else:
        fig, ax = plt.subplots()
        ax.plot(lpsi.time_series_motion.to_pystrata.times,
                lpsi.time_series_motion.to_pystrata.accels)
        ax.set(xlabel='time (s)', ylabel='acceleration (g)')
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(f'### Response spectrum - {100*results_damping:.1f}% damping')
    resp_spec = lproject.response_spectrum(damping=results_damping)
    if inteactive_charts:
        p=figure(x_axis_type="log", y_axis_type="log", x_axis_label='Frequency (Hz)', y_axis_label='Spectral Acceleration (g)')
        p.line(resp_spec.freqs, resp_spec.values)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)

    else:
        st.pyplot(resp_spec.plot().get_figure())


############################################################################################################
# Soil profile
############################################################################################################

# Main
st.markdown('## Soil profile')

st_soil_profile_main_expander = st.expander(
    label='Soil profile', expanded=False)
with st_soil_profile_main_expander:

    st.pyplot(lpsi.to_pystrata_profile.plot("initial_shear_vel").get_figure())
    # st.pyplot(lpsi.to_pystrata_profile.plot("shear_vel").get_figure())





# profile = pystrata.site.Profile(list_layers).auto_discretize()


# st.markdown('#### ***Time Series Motion Image Plot***')

# fig, ax = plt.subplots()
# ax.plot(ts.times, ts.accels)
# ax.set(xlabel='time (s)', ylabel='acceleration (g)')
# fig.tight_layout()
# st.pyplot(fig)

# motion = pystrata.motion.TimeSeriesMotion(
#     f'{uploaded_file}', description='', time_step=ts.time_step, accels=ts.accels)


# input_data['strain_limit'] = st.sidebar.number_input(
#     label='strain limit', value=input_data['strain_limit'], format='%.3f')
# input_data['damping'] = st.sidebar.number_input(
#     label='damping', value=input_data['damping'], format='%.3f')


# calc = pystrata.propagation.EquivalentLinearCalculator(
#     strain_limit=input_data['strain_limit'])

# freqs = np.logspace(-1, 2, num=500)

# outputs = pystrata.output.OutputCollection(
#     [
#         pystrata.output.ResponseSpectrumOutput(
#             # Frequency
#             freqs,
#             # Location of the output
#             pystrata.output.OutputLocation("outcrop", index=0),
#             # Damping
#             input_data['damping'],
#         ),
#         pystrata.output.ResponseSpectrumRatioOutput(
#             # Frequency
#             freqs,
#             # Location in (denominator),
#             pystrata.output.OutputLocation("outcrop", index=-1),
#             # Location out (numerator)
#             pystrata.output.OutputLocation("outcrop", index=0),
#             # Damping
#             input_data['damping'],
#         ),
#         pystrata.output.MaxStrainProfile(),
#     ]
# )


# calc(motion, profile, profile.location("outcrop", index=-1))
# outputs(calc)
# for o in outputs:
#     f, ax = plt.subplots()
#     ax = o.plot(style="indiv")

#     st.pyplot(f)


# df = outputs[0].to_dataframe()

# freqs = df.index.to_numpy()
# sas = np.flip(df['r1'].to_numpy())
# periods = np.flip(1/freqs)

# fig, ax = plt.subplots()
# ax.plot(periods, sas)
# ax.set(xlabel='period (s)', ylabel='spectral acceleration (g)')
# fig.tight_layout()

# st.pyplot(fig)


# st.write(input_data['strain_limit'])


############################################################################################################
# Results
############################################################################################################
st.markdown('## Results')

st_results_main_expander = st.expander(label='Results', expanded=False)
with st_results_main_expander:

    st.markdown('#### Response Spectrum - bedrock vs ground surface')

    # Στην επιφάνεια
    out_index0 = lproject.response_spectrum(
        freqs=freqs, damping=results_damping, location_index=0)
    # Στον βράχο
    out_index1 = lproject.response_spectrum(
        freqs=freqs, damping=results_damping, location_index=-1)

    if inteactive_charts:
        p=figure(x_axis_label='Period (g)', y_axis_label='Spectral Acceleration (g)')
        p.line(out_index0.periods, out_index0.values, legend_label='Ground surface', color='red')
        p.line(out_index1.periods, out_index1.values, legend_label='Bedrock', color='blue')
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)
    else:
        fig_results_spectra, ax_results_spectra = plt.subplots()
        ax_results_spectra.plot(
            out_index0.periods, out_index0.values, linewidth=1.0, color='red', label='Ground surface')
        ax_results_spectra.plot(
            out_index1.periods, out_index1.values, linewidth=1.0, color='blue', label='Bedrock')
        ax_results_spectra.set(xlabel='Period (s)', ylabel='Spectral Acceleration (g)')
        ax_results_spectra.legend()
        st.pyplot(fig_results_spectra)

    st.markdown('#### Response spectrum')
    st.pyplot(lproject.response_spectrum(location_index=-1,
              damping=results_damping).plot().get_figure())

    st.markdown('#### Transfer function')
    st.pyplot(lproject.accel_transfer_function().plot().get_figure())

    st.markdown('#### Response_spectrum_ratio')
    st.pyplot(lproject.response_spectrum_ratio().plot().get_figure())

    st.markdown('#### Fourier amplitude spectrum')
    st.pyplot(lproject.fourier_amplitude_spectrum().plot().get_figure())



