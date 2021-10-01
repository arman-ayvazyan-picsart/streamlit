import math
import tempfile
import numpy as np
import pandas as pd
from os import listdir
import streamlit as st
from methods import video_stream
from os.path import isfile, join


st.set_page_config(
    page_title="Frame Grouping - Interactive 2.0",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(f'My first app {st.__version__}')

st.title('Frame Grouping - Interactive 2.0')
method_list = ["Main", "Light only"]

s1_col1, s1_col2 = st.columns(2)
with s1_col1:
  st.subheader("Select Method")
  form_method = st.selectbox(
      'What method to use?',
      method_list)

with s1_col2:
  st.subheader("Upload a video")
  uploaded_file = st.file_uploader("Which video to process?")

form_adaptive = st.checkbox("Apply adaptive threshold", value=True)
if not form_adaptive:
    form_manual = st.checkbox("Manually define the threshold?", value=False)
    if form_manual:
        form_threshold = st.slider('Select the threshold to apply. (all values in % s)', 0.0, 100.0, 99.0, 0.01)
form_sliding_window = st.slider('Select the minimum frames per clip. (Shorter clips will be merged)', 0, 100, 60, 1)

#stframe = st.empty()
#stframe.image(gray)

if uploaded_file is not None:
    # To read file as bytes:
    # bytes_data = uploaded_file.getvalue()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    form_video = tfile.name

if st.button('Start the process'):
    frame_ids, times, frame_diff, groups = video_stream(form_video, form_method, form_adaptive, form_sliding_window)

    st.subheader('Results:')

    '- **Method: **', form_method
    if not form_adaptive:
        '- **Threshold: **', form_threshold
    '- **Sliding window: **', form_sliding_window, " *frames*"
    '- **Video Duration: **', times[0], " *seconds*"

    '**Runtime Durations: **'
    '- **Method: **', times[1], " *seconds*"
    '- **Threshold: **', times[2], " *seconds*"

    '**Frame IDs of clip cuts: **'
    frame_ids

    st.download_button(label="Download the video", data=form_video, file_name='result.mp4', mime='video/mp4')
