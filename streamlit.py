import math
import tempfile
import numpy as np
import pandas as pd
from os import listdir
import streamlit as st
from methods import video_file, video_stream
from os.path import isfile, join


st.set_page_config(
    page_title="Frame Grouping - Interactive 2.0",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Frame Grouping - Interactive 2.0')

source_video_list = [f for f in listdir("./source/") if isfile(join("./source/", f))]
method_list = ["Main", "Light only"]

s1_col1, s1_col2 = st.beta_columns(2)
with s1_col1:
  st.subheader("Select Method")
  form_method = st.selectbox(
      'What method to use?',
      method_list)

with s1_col2:
  st.subheader("Select Video")
  form_video = st.selectbox(
      'Which video to process?',
      source_video_list)

uploaded_file = st.file_uploader("Or upload a custom video")

form_adaptive = st.checkbox("Apply adaptive threshold")
if not form_adaptive:
    form_threshold = st.slider('Select the threshold to apply. (all values in % s)', 0, 100, 80, 1)
form_sliding_window = st.slider('Select the minimum frames per clip. (Shorter clips will be merged)', 0, 100, 15, 1)

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

    '- **Video: **', form_video
    '- **Method: **', form_method
    '- **Threshold: **', form_threshold
    '- **Min Frames: **', form_sliding_window
    '- **Video Duration: **', times[0], " *seconds*"

    '**Runtime Durations: **'
    '- **Method: **', times[1], " *seconds*"
    '- **Merge: **', times[2], " *seconds*"

    '**Frame IDs of clip cuts: **'
    '- **Method: **', frame_ids, " *seconds*"
