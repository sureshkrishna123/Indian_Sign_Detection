from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np

from keras.models import load_model

model = load_model('/content/demo_model.h5')

demo_image_path = '/content/drive/MyDrive/dataset/archive/test/S/101.jpg'
img = tf.keras.utils.load_img(
    demo_image_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {}.".format(class_names[np.argmax(score)]))
