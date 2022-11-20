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


FREQS = np.logspace(-0.5, 2, num=500)

################################################
# Session state initialization
################################################
if 'linput' not in st.session_state:
    st.session_state['linput'] = LPyStrataInput.from_json_file(
        'streamlit/ini.json')

if 'lproject' not in st.session_state:
    st.session_state['lproject'] = LProject(st.session_state['linput'])

if 'interactive_charts' not in st.session_state:
    st.session_state['interactive_charts'] = False

if 'interactive_charts_theme' not in st.session_state:
    st.session_state['interactive_charts_theme'] = 'light_minimal'

if 'motion_description' not in st.session_state:
    st.session_state['motion_description'] = ''


################################################
# Intro text
################################################

st.title('1-D Seismic Site Response Analysis')

st.markdown('This is a web application for one-dimensional site response analysis using linear elastic and equivalent linear analysis.')

st.image('streamlit/img/DaPan.png', width=650)
st.markdown('[Source: DOI: 10.1785/0120210300](https://pubs.geoscienceworld.org/ssa/bssa/article/doi/10.1785/0120210300/612887/Deep-Neural-Network-Based-Estimation-of-Site)')


# ******************************************************************
# Sidebar / Files - Input parameters - Options
# ******************************************************************

###################################
# Sidebar / Project input file
###################################
with st.sidebar.expander(label='Import/export input file', expanded=False):

    # Save json input file
    st.download_button('Save input file', LPyStrataInput.schema().dumps(
        st.session_state['linput']), 'lampa.json', 'application/json')

    st.write('---')
    # Load json input file
    uploaded_input = st.file_uploader('Load input file', type=['json'])
    if uploaded_input is not None:
        file_details = {"FileName": uploaded_input.name,
                        "FileType": uploaded_input.type, "FileSize": uploaded_input.size}

        utf_read = uploaded_input.read().decode('utf-8')
        st.session_state['linput'] = LPyStrataInput.schema().loads(utf_read)

st.sidebar.markdown('---')

###################################
# Sidebar / Seismic Motion
###################################


def update_motion_description():
    st.session_state['linput'].time_series_motion.description = st.session_state['motion_description']


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
        ltsm.description = uploaded_accel.name

        st.session_state['motion_description'] = ltsm.description
        st.session_state['linput'].time_series_motion = ltsm

    st.markdown('### Motion description')
    st.text_input(label='Motion description',
                  key='motion_description', on_change=update_motion_description)

    st.markdown(f"{st.session_state['linput'].time_series_motion.description}")


st.sidebar.markdown('---')


###################################
# Sidebar / Soil Layers
###################################

# st.sidebar.markdown('## Soil Layers')
with st.sidebar.expander(label='Soil Layers', expanded=False):

    layers = st.session_state['linput'].layers

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

###################################
# Sidebar / Calculator
###################################

st_calculator_sidebar_expander = st.sidebar.expander(
    label='Calculator', expanded=False)
with st_calculator_sidebar_expander:
    calc_type = st.radio('Select calculator',
                         options=['LinearElasticCalculator',
                                  'EquivalentLinearCalculator'],
                         key=st.session_state['linput'].calculator_type)
    st.session_state['linput'].calculator_type = calc_type

st.sidebar.markdown('---')

###################################
# Sidebar / Results
###################################

# st.sidebar.markdown('## Results')
st_results_sidebar_expander = st.sidebar.expander(
    label='Results options', expanded=False)
with st_results_sidebar_expander:
    results_damping = st.number_input(
        label='damping', value=0.05, format='%.3f')


st.sidebar.markdown('---')

###################################
# Sidebar / General Options
###################################

# st.sidebar.markdown('## General Options')
st_options_sidebar_expander = st.sidebar.expander(
    label='General Options', expanded=False)
with st_options_sidebar_expander:
    st.checkbox(label='Interactive charts', key='interactive_charts')
    st.selectbox(label='Interactive chart theme',
                 options=['caliber', 'dark_minimal', 'light_minimal',
                          'night_sky', 'contrast'],
                 key='interactive_charts_theme')

    doc = curdoc()  # https://discuss.streamlit.io/t/bokeh-theming/15302
    doc.theme = st.session_state['interactive_charts_theme']


##################################################################################
# ********************************************************************************
# ********** Main view ***********************************************************
# ********************************************************************************
##################################################################################


st.markdown('## Project information')
# linput.name = st.text_input(label='Project name', value=linput.name)

st.write()


###########################
# Main / Seismic Motion
###########################

st.markdown('## Seismic motion')

st_seismic_motion_main_expander = st.expander(
    label='Seismic motion', expanded=False)
with st_seismic_motion_main_expander:

    st.markdown('### Accelerations')
    if st.session_state['interactive_charts']:
        p = figure(x_axis_label='Time (sec)',
                   y_axis_label='Acceleration (g)')
        p.line(st.session_state['linput'].time_series_motion.to_pystrata.times,
               st.session_state['linput'].time_series_motion.to_pystrata.accels)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)

    else:
        fig, ax = plt.subplots()
        ax.plot(st.session_state['linput'].time_series_motion.to_pystrata.times,
                st.session_state['linput'].time_series_motion.to_pystrata.accels)
        ax.set(xlabel='time (s)', ylabel='acceleration (g)')
        fig.tight_layout()
        st.pyplot(fig)

    st.markdown(f'### Response spectrum - {100*results_damping:.1f}% damping')
    resp_spec = st.session_state['lproject'].response_spectrum(
        damping=results_damping)
    if st.session_state['interactive_charts']:
        p = figure(x_axis_type="log", y_axis_type="log",
                   x_axis_label='Frequency (Hz)',
                   y_axis_label=f'{100*results_damping:.1f}% Damped, Spectral Acceleration (g)')
        p.line(resp_spec.freqs, resp_spec.values)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)

    else:
        st.pyplot(resp_spec.plot().get_figure())


############################################################################################################
# Main / Soil profile
############################################################################################################

st.markdown('## Soil profile')

st_soil_profile_main_expander = st.expander(
    label='Soil profile', expanded=False)
with st_soil_profile_main_expander:

    st.pyplot(st.session_state['linput'].to_pystrata_profile.plot(
        "initial_shear_vel").get_figure())
    # st.pyplot(linput.to_pystrata_profile.plot("shear_vel").get_figure())


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
# Main / Results
############################################################################################################
st.markdown('## Results')

st_results_main_expander = st.expander(label='Results', expanded=False)
with st_results_main_expander:

    ############################################################################
    st.markdown('#### Response Spectrum - bedrock vs ground surface')
    # Στην επιφάνεια
    response_spectrum_surface = st.session_state['lproject'].response_spectrum(
        freqs=FREQS, damping=results_damping, location_index=0)
    # Στον βράχο
    response_spectrum_bedrock = st.session_state['lproject'].response_spectrum(
        freqs=FREQS, damping=results_damping, location_index=-1)

    if st.session_state['interactive_charts']:
        p = figure(x_axis_label='Period (sec)',
                   y_axis_label=f'{100*results_damping:.1f}% Damped, Spectral Acceleration (g)')
        p.line(response_spectrum_surface.periods, response_spectrum_surface.values,
               legend_label='Ground surface', color='red')
        p.line(response_spectrum_bedrock.periods, response_spectrum_bedrock.values,
               legend_label='Bedrock', color='blue')
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)
    else:
        fig_results_spectra, ax_results_spectra = plt.subplots()
        ax_results_spectra.plot(
            response_spectrum_surface.periods, response_spectrum_surface.values, linewidth=1.0, color='red', label='Ground surface')
        ax_results_spectra.plot(
            response_spectrum_bedrock.periods, response_spectrum_bedrock.values, linewidth=1.0, color='blue', label='Bedrock')
        ax_results_spectra.set(xlabel='Period (sec)',
                               ylabel=f'{100*results_damping:.1f}% Damped, Spectral Acceleration (g)')
        ax_results_spectra.legend()
        st.pyplot(fig_results_spectra)

    ############################################################################
    st.markdown('#### Response spectrum - Frequency')
    if st.session_state['interactive_charts']:
        p = figure(x_axis_type="log", y_axis_type="log",
                   x_axis_label='Frequency (Hz)',
                   y_axis_label=f'{100*results_damping:.1f}% Damped, Spectral Acceleration (g)')
        p.line(response_spectrum_surface.freqs,
               response_spectrum_surface.values)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)
    else:
        st.pyplot(response_spectrum_surface.plot().get_figure())

    ############################################################################
    st.markdown('#### Transfer function')
    accel_transfer_function = st.session_state['lproject'].accel_transfer_function(
        freqs=FREQS)
    if st.session_state['interactive_charts']:
        p = figure(x_axis_type="log", y_axis_type="log",
                   x_axis_label='Frequency (Hz)', y_axis_label='Accel. Transfer Function')
        p.line(accel_transfer_function.freqs, accel_transfer_function.values)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)
    else:
        st.pyplot(accel_transfer_function.plot().get_figure())

    ############################################################################
    st.markdown('#### Response_spectrum_ratio')
    response_spectrum_ratio = st.session_state['lproject'].response_spectrum_ratio(
        freqs=FREQS, damping=results_damping)
    if st.session_state['interactive_charts']:
        p = figure(x_axis_type="log", y_axis_type="log",
                   x_axis_label='Frequency (Hz)',
                   y_axis_label=f'{100*results_damping:.1f}% Damped, Resp. Spectral Ratio')
        p.line(response_spectrum_ratio.freqs, response_spectrum_ratio.values)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)
    else:
        st.pyplot(response_spectrum_ratio.plot().get_figure())

    ############################################################################
    st.markdown('#### Fourier amplitude spectrum')
    fourier_amplitude_spectrum = st.session_state['lproject'].fourier_amplitude_spectrum(
        freqs=FREQS)
    if st.session_state['interactive_charts']:
        p = figure(x_axis_type="log", y_axis_type="log",
                   x_axis_label='Frequency (Hz)',
                   y_axis_label='Fourier Ampl. (cm/s)')
        p.line(fourier_amplitude_spectrum.freqs,
               fourier_amplitude_spectrum.values)
        doc.add_root(p)
        st.bokeh_chart(p, use_container_width=True)
    else:
        st.pyplot(fourier_amplitude_spectrum.plot().get_figure())


st.sidebar.write(st.session_state)
