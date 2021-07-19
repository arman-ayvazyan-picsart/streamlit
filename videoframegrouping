st.title('Video Frame Grouping Interactive Showcase')

st.subheader('Method')
selected_method = st.select_slider( 'Select the method', options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'], value='red')
st.write('You selected method', selected_method)

st.subheader('Video')
selected_video = st.select_slider( 'Select the video', options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'], value='red')
st.write('You selected video ', selected_video)

st.subheader('This is a subheader')
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['a', 'b', 'c'])
st.line_chart(chart_data)


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
