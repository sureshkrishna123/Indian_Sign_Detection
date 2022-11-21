

st.set_page_config(layout="wide")
st.sidebar.image('images/Azure_Image.png', width=300)
st.sidebar.header('A website for classifying Indian Sign Language')
st.sidebar.markdown('Used CNN algorithm')


app_mode = st.sidebar.radio(
    "",
    ("About Me","Indian Sign Language"),
)


st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('N.V.Suresh Krishna | sureshkrishnanv24@gmail.com https://github.com/sureshkrishna123')

if app_mode =='About Me':
    st.image('images/wp4498220.jpg', use_column_width=True)
    st.markdown('''
              # About Me \n 
                Hey this is ** N.V.Suresh Krishna **. \n
                
                
                Also check me out on Social Media
                - [git-Hub](https://github.com/sureshkrishna123)
                - [LinkedIn](https://www.linkedin.com/in/suresh-krishna-nv/)
                - [Instagram](https://www.instagram.com/worldofsuresh._/)
                - [Portfolio](https://sureshkrishna123.github.io/sureshportfolio/)
                - [Blog](https://ingenious-point.blogspot.com/)\n
                If you are interested in building more about Microsoft Azure then   [click here](https://azure.microsoft.com/en-in/)\n
                ''')
               

if app_mode=='Indian Sign Language':
  st.image(os.path.join('./images','facial-recognition-software-image.jpg'),use_column_width=True )
  st.title("Final year project")
  st.header('Indian Sign Language Classification')
  st.markdown("Using CNN algorithm, the hand sign images are classified and gives the text as an output.")
  st.text("")
  
  image_file =  st.file_uploader("Upload Images (less than 1mb)", type=["png","jpg","jpeg"])
  if image_file is not None:
    img = Image.open(image_file)
    st.image(image_file,width=250,caption='Uploaded image')
    byte_io = BytesIO()
    img.save(byte_io, 'PNG')#PNG
    image = byte_io.getvalue()


  button_translate=st.button('Click me',help='To give the image')

  if button_translate and image_file :

    model = load_model('/content/demo_model.h5')
    demo_image_path = image_file
    img = tf.keras.utils.load_img(demo_image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    st.text("This image most likely belongs to "+class_names[np.argmax(score)])
    
