from fileinput import filename
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pystrata
import pyexcel
import json

from pystrata.motion import TimeSeriesMotion
from pystrata.site import Profile
from PIL import Image

from fileloaders import time_accel_txt_to_pystrata_motion

from lampa.input import PyStrataInput


st.title('1-D Seismic Site Response Analysis')

st.text('This is a web app for one dimensional ground response analysis \nusing data from seismic motions for the development of \nAcceleration Response Spectra and Spectral Ratio \nPystrata library etc etc')

psi = PyStrataInput(name = 'test1',
                    time_series_motion={'a': np.array([1]), 't': np.array([2])})

st.write(psi)


# load ini json file
ini_file = r'ini.json'
with open(ini_file, 'r') as f:
    input_data = json.load(f)

st.write(input_data)


############################################################################################################
# STEP 1: Load accelerogram
############################################################################################################
st.sidebar.markdown('### STEP 1: Load accelerogram')
with st.sidebar.expander(label='Input accelerogram', expanded=False):



    accel_filetype = st.radio('Select filetype', options=['xmlx', 'txt', 'csv', 'AT2'])

    uploaded_file = st.file_uploader('Upload your file here')

    if uploaded_file is not None:

        input_data['accel_file'] = uploaded_file.name
        ts = time_accel_txt_to_pystrata_motion(uploaded_file)
    else:
        ts = time_accel_txt_to_pystrata_motion(input_data['accel_file'])


#image = Image.open ('1d.jpg')
#st.image(image, use_column_width=True)










st.markdown('#### ***Time Series Motion Image Plot***')

fig, ax = plt.subplots()
ax.plot(ts.times, ts.accels)
ax.set(xlabel='time (s)', ylabel='acceleration (g)')
fig.tight_layout()
st.pyplot(fig)

motion = pystrata.motion.TimeSeriesMotion(
    f'{uploaded_file}', description='', time_step=ts.time_step, accels=ts.accels)


input_data['strain_limit'] = st.sidebar.number_input(
    label='strain limit', value=input_data['strain_limit'], format='%.3f')
input_data['damping'] = st.sidebar.number_input(
    label='damping', value=input_data['damping'], format='%.3f')

st.sidebar.header('STEP 2: Define Soil Layers')
with st.sidebar.expander(label='Soil Layers', expanded=False):
    ############################################################################################################
    # STEP 2: Select Soil Layers
    ############################################################################################################

    no_layers = st.slider('Number of Layers', 1, 20, 1)

    layer_names = [f'Layer {i}' for i in range(1, no_layers+1)]

    layers = []

    layer_tabs = st.tabs(layer_names)

    for i in range(0, no_layers):
        with layer_tabs[i]:
            layer = {}
            layer_tabs[i].header(f'Layer {i+1} properties')
            layer['unit_wt'] = layer_tabs[i].number_input(
                key=f'unit_wt {i+1}', label='unit weight of the material [kN/m³]', value=18.0)
            layer['plas_index'] = layer_tabs[i].number_input(
                key=f'plas_index {i+1}', label='plasticity index [percent]', value=0.0)
            layer['ocr'] = layer_tabs[i].number_input(
                key=f'ocr {i+1}', label='over-consolidation ratio', value=1)
            layer['stress_mean'] = layer_tabs[i].number_input(
                key=f'stress_mean {i+1}', label='mean effective stress [kN/m²]', value=200)
            layer['freq'] = layer_tabs[i].number_input(
                key=f'freq {i+1}', label='excitation frequency [Hz]', value=30)
            layer['num_cycles'] = layer_tabs[i].number_input(
                key=f'num_cycles {i+1}', label='number of cycles of loading', value=500)
            # layer_tabs[i].number_input(key=f'strains {i+1}', label = 'shear strains levels [decima', value=18.0)
            layers.append(layer)

# st.write(layers)

# selected_layer = st.select_slider('select layer', options=range(1, no_layers+1))
# st.write(f'selected_layer: {selected_layer}')
# st.write(f'unit_wt for selected_layer: {layers[selected_layer-1]["unit_wt"]}')


list_layers = []

list_layers.append(pystrata.site.Layer(
    pystrata.site.DarendeliSoilType(16.0, plas_index=0, ocr=1, stress_mean=40),
    5,
    140,
))

list_layers.append(pystrata.site.Layer(
    pystrata.site.DarendeliSoilType(
        18.0, plas_index=25, ocr=1, stress_mean=215),
    15,
    250,
))

list_layers.append(pystrata.site.Layer(
    pystrata.site.DarendeliSoilType(
        20.0, plas_index=0, ocr=1, stress_mean=650),
    30,
    450,
))

list_layers.append(pystrata.site.Layer(
    pystrata.site.SoilType("Rock", 23.0, None, 0.02), 0, 1200))


profile = pystrata.site.Profile(list_layers).auto_discretize()


calc = pystrata.propagation.EquivalentLinearCalculator(
    strain_limit=input_data['strain_limit'])

freqs = np.logspace(-1, 2, num=500)

outputs = pystrata.output.OutputCollection(
    [
        pystrata.output.ResponseSpectrumOutput(
            # Frequency
            freqs,
            # Location of the output
            pystrata.output.OutputLocation("outcrop", index=0),
            # Damping
            input_data['damping'],
        ),
        pystrata.output.ResponseSpectrumRatioOutput(
            # Frequency
            freqs,
            # Location in (denominator),
            pystrata.output.OutputLocation("outcrop", index=-1),
            # Location out (numerator)
            pystrata.output.OutputLocation("outcrop", index=0),
            # Damping
            input_data['damping'],
        ),
        pystrata.output.MaxStrainProfile(),
    ]
)


calc(motion, profile, profile.location("outcrop", index=-1))
outputs(calc)
for o in outputs:
    f, ax = plt.subplots()
    ax = o.plot(style="indiv")

    st.pyplot(f)


df = outputs[0].to_dataframe()

freqs = df.index.to_numpy()
sas = np.flip(df['r1'].to_numpy())
periods = np.flip(1/freqs)

fig, ax = plt.subplots()
ax.plot(periods, sas)
ax.set(xlabel='period (s)', ylabel='spectral acceleration (g)')
fig.tight_layout()

st.pyplot(fig)


st.write(input_data['strain_limit'])
