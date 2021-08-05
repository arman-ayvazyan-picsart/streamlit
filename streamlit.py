import math
import numpy as np
import pandas as pd
from os import listdir
import streamlit as st
from methods import method
from os.path import isfile, join


st.set_page_config(
    page_title="Frame Grouping - Interactive 1.0",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title('Frame Grouping - Interactive 1.0')

source_video_list = [f for f in listdir("./source/") if isfile(join("./source/", f))]
method_list = ["Hash", "Lab", "Intersection", "SSIM"]

s1_col1, s1_col2 = st.beta_columns(2)
with s1_col1:
  st.subheader("Select Method")
  selected_method = st.selectbox(
      'What method to use?',
      method_list)

with s1_col2:
  st.subheader("Select Video")
  selected_video = st.selectbox(
      'Which video to process?',
      source_video_list)

selected_threshold = st.slider('Select the threshold to apply. (all values in % s)', 0, 100, 80, 1)
selected_min_clip_size = st.slider('Select the minimum frames per clip. (Shorter clips will be merged)', 0, 100, 15, 1)

r1, r2, r3, times, output = method(selected_video, selected_threshold, selected_min_clip_size, selected_method)

r0 = np.ones(r1.shape)*selected_threshold
chart_data = pd.DataFrame(np.transpose(np.vstack((r0, r1))), columns=['Threshold', 'Method'])

st.subheader('Difference')
st.line_chart(chart_data)

s3_col1, s3_col2 = st.beta_columns(2)
with s3_col1:
    st.subheader('Histogram')
    cmin = min(r1)
    cmax = max(r1)
    hist_values = np.histogram(r1, bins=20, range=(min(cmin, 0), math.ceil(cmax)))[0]
    st.bar_chart(hist_values)
with s3_col2:
    st.subheader('Groups')
    st.line_chart(r2)

video_file1 = open("./source/"+selected_video, 'rb')
video_bytes1 = video_file1.read()

#video_file2 = open("./result/"+selected_video.rstrip('.mp4')+".webM", 'rb')
video_file2 = open(output, 'rb')
video_bytes2 = video_file2.read()
  
st.subheader('Results:')

'- **Video: **', selected_video
'- **Method: **', selected_method
'- **Threshold: **', selected_threshold
'- **Min Frames: **', selected_min_clip_size
'- **Video Duration: **', times[0], " *seconds*"

'**Runtime Durations: **'
'- **Method: **', times[1], " *seconds*"
'- **Merge: **', times[2], " *seconds*"
'- **Export: **', times[3], " *seconds*"

s2_col1, s2_col2 = st.beta_columns(2)
with s2_col1:
  st.subheader("Original")
  st.video(video_bytes1)

with s2_col2:
  st.subheader("Processed")
  st.video(video_bytes2)
