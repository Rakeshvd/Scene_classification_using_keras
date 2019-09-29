import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time
import matplotlib.pyplot as plt

#Define Path
model_path = '/content/logs/model.h5'
model_weights_path = '/content/logs/weights.h5'
test_path = '/content/keras_implementation_small/test'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 150, 150

#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  plt.show(x)
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  print(result)
  answer = np.argmax(result)
  if answer == 0:
    print("Predicted:buildings")
  elif answer == 1:
    print("Predicted:forest")
  elif answer == 2:
    print("Predicted:glacier")
  elif answer == 3:
    print("Predicted:street")
  

  return answer

#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(" ")
