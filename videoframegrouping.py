import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('Video Frame Grouping Interactive Showcase')


df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})


st.subheader('Method')
selected_method = st.selectbox(
    'What method to use?',
     df['first column'])

'You selected: ', selected_method

st.subheader('Video')
selected_video = st.selectbox(
    'Which video to process?',
     df['first column'])

'You selected: ', selected_video

st.subheader('This is a subheader')
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.line_chart(chart_data)


selected_threshold = st.slider('Select the threshold to apply?', 0, 130, 25)
st.write('Threshold: ', selected_threshold)

video_file1 = open('myvideo.mp4', 'rb')
video_bytes1 = video_file1.read()
  
video_file2 = open('myvideo.mp4', 'rb')
video_bytes2 = video_file2.read()
  
st.subheader('Results')
col1, col2, col3 = st.beta_columns(3)
with col1:
  st.header("Original")  
  st.video(video_bytes)

with col1:
  st.header("Processed")  
  st.video(video_bytes)
