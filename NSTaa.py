# -*- coding: utf-8 -*-

# =============================================================================
# Importing Necessary Libraries
# =============================================================================
from PIL import Image
import streamlit as st
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False
import numpy as np
import pandas as pd
import time
import altair as alt
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models 
import datetime


# If previous User forgot to delete his data
ImageNames123 = os.listdir()
if "Your-Style-Image.jpg" in ImageNames123:
    os.remove("Your-Style-Image.jpg")

if "Your-Content-Image.jpg" in ImageNames123:
    os.remove("Your-Content-Image.jpg")

"""## Neural Style Transfer
In this post, I will be introducing a tool I developed based on an algorithm [**'Neural Style Transfer'**](https://arxiv.org/abs/1508.06576)!
Feel free to apply it to your data.

**N**eural **S**tyle **T**ransfer is a technique to extract features i.e.the appearance or visual style, from the 
reference image also known as **'Style Image'** and to apply it to your image. NST algorithms have been implemented via deep 
neural networks. A common application of **'NST'** is a creation of artificial work from photographs/paintings by transferring
the features of the paintings to the user-supplied photographs i.e, the **'Content Image'**. 
Here, is an interesting [**article**](https://www.christies.com/features/A-collaboration-between-two-artists-one-human-one-a-machine-9332-1.aspx)
stating the application of the same.

Below is an example that was generated using NST. Here, the content image is 
taken digitally somewhere in Austria and the style image ([**Starry Night**](https://www.vincentvangogh.org/starry-night.jsp))
 is from famous artist **Van Gogh**. Now imagine what happens if Van Gogh tried to paint the content image, do you
know how would it look like? The below image shows it would look something like this.  

"""
st.image(Image.open("linkedin.jpg"), width=512)


def returnNotMatches(a, b):
    return [x for x in a if x not in b]

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

# datetime object containing current date and time
now = datetime.datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("date and time =", dt_string)

# =============================================================================
# Upload User Style/Content image
# =============================================================================

uploaded_file = st.file_uploader("Choose a Content Image file", type=["jpg","jpeg","png", "JPEG"])

if uploaded_file is not None:
    data = Image.open(uploaded_file)
    data = data.convert("RGB")
    data.save('Your-Content-Image.jpg')
    # st.image(data, use_column_width=True)

uploaded_file1 = st.file_uploader("Choose a Style Image file", type=["jpg","jpeg","png", "JPEG"])
if uploaded_file1 is not None:
    data1 = Image.open(uploaded_file1)
    data1 = data1.convert("RGB")
    data1.save('Your-Style-Image.jpg')
    # st.image(data, use_column_width=True)
    st.empty()

# =============================================================================
# Dropdown list of Style and Content Image
# =============================================================================

ImageNames123 = os.listdir()
# st.write(ImageNames123)
contentImageNames = ["House-in-Austria.jpg", "River.JPG", "King-of-Walhalla.JPG",
                        "River-Bridge-Path.JPG", "River-Bridge-Tree.JPG", "Riverfront.JPG", "Riverfront-Cycle.JPG",
                        "Tree-Springs.JPG", "Tree-Springs1.JPG", "View-From-Walhalla.JPG", "Walhalla-Regensburg.jpg"]

if "Your-Content-Image.jpg" in ImageNames123:
    contentImageNames = ["House-in-Austria.jpg", "River.JPG", "King-of-Walhalla.JPG",
                        "River-Bridge-Path.JPG", "River-Bridge-Tree.JPG", "Riverfront.JPG", "Riverfront-Cycle.JPG",
                        "Tree-Springs.JPG", "Tree-Springs1.JPG", "View-From-Walhalla.JPG", "Walhalla-Regensburg.jpg", "Your-Content-Image.jpg"]
else:
    pass
contentImage = st.selectbox("Please Select a Content Picture", contentImageNames, len(contentImageNames)-1)
selectedContentImage = contentImage

styleImageNames = ["La-Mousme.jpg", "Self-Potrait.jpg", "Starry-Night.jpg", "Tuebingen-Neckarfront.jpg",
      "Vassily-Kandinsky.jpg", "Starry-Night-Over-the-Rhone.jpg","Style-Art-Image.jpg", "Waves.jpg"]

if "Your-Style-Image.jpg" in ImageNames123:
    styleImageNames = ["La-Mousme.jpg", "Self-Potrait.jpg", "Starry-Night.jpg", "Tuebingen-Neckarfront.jpg",
      "Vassily-Kandinsky.jpg", "Starry-Night-Over-the-Rhone.jpg","Style-Art-Image.jpg", "Waves.jpg", "Your-Style-Image.jpg"]
else:
    pass
styleImage = st.selectbox("Please Select a Style Picture", styleImageNames, len(styleImageNames)-1)
selectedStyleImage = styleImage

# Set up some global values here
content_path = selectedContentImage
style_path = selectedStyleImage

# =============================================================================
# Loading and Scalling Image
# =============================================================================
def load_img(path_to_img):

  max_dim = 512
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
  
  img = kp_image.img_to_array(img)
  
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img
# =============================================================================
# Reduce the dimension of image as well as Normalizing for display
# =============================================================================
def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)
  # st.image(out)

"""These are input content and style images. We hope to **"create"** an image with the content
 of our content image, but with the style of the style image."""


# =============================================================================
# Pre process image for VGG16 model
# =============================================================================

def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg16.preprocess_input(img)
  return img

# =============================================================================
# Deprocess image for plotting image
# =============================================================================

def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = np.squeeze(x, 0)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [1, height, width, channel] or [height, width, channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x

# =============================================================================
# Predifined/Default Style and Content Layers
# =============================================================================

# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


"""### Intuitive Explanation of the NST Technique
As part of the implementation of NST I am using [VGG16](https://keras.io/applications/#vgg16) pretrained network for our custom images. 
And VGG networks are trained on an image with each channel normalized by `mean = [103.939, 116.779, 123.68]`
Hence, we also need to normalize our input images as per the VGG16 model standard.

The below figure shows `model.summary()` of VGG16 model:
"""
image = Image.open('vgg16.png')
st.image(image, caption='VGG16 Model Summary',use_column_width=True)

"""
As we go deeper into the layers the features extracted from the image, increases.
The last layer of VGG16 model has the highest number of features i.e. shapes. And the shallow layers
contains a small number of extracted features which is useful to apply on our style image to extract 
vital pieces of information like the stroke of the brush have happened into the style image to adopt.
Hence, our Content image is only trained on the last layer of VGG16 and our Style image is trained on 
all the layers except the last layer of VGG16.

Here, for your image you can choose which layer to consider for Content image and Style image.
You can try different combinations and can compare the results later on. As well as one can use 
this tool for hyperparameter tuning.
"""

# =============================================================================
# Defining and loading VGG16 Model
# =============================================================================
def get_model():

  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet')
  # vgg = VGG16(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)

moDel = get_model()
ln = []
for layer in moDel.layers:
    ln.append(layer.name)
content_image = load_and_process_img(content_path)
style_image = load_and_process_img(style_path)

style_layers = st.multiselect('Please select layer/s to train Style image on', ln[1:], default=style_layers)
content_layers = st.multiselect('Please select layer/s to train Content image on', ln[1:], default=content_layers)



num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# =============================================================================
# Computing Content Loss
# =============================================================================
def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))

content_sidebar = st.sidebar.empty()
style_sidebar = st.sidebar.empty()

content_sidebar.image(Image.open(selectedContentImage), caption="Content Image", width=200)

style_sidebar.image(Image.open(selectedStyleImage), caption="Style Image", width=200)

"""
In NST the loss function is the summation of loss between input image with the content image (content loss)
 and loss between an input image and Style image (style loss).
And also different weightage for content loss and style loss are given to optimize respective losses more.
"""

# =============================================================================
# Hyperparameter Selection
# =============================================================================
content_weight = st.number_input('Please Insert Content Weight', value=1e3)
st.write('Selected Content Weight is: {}'.format(content_weight))
style_weight = st.number_input('Please Insert Style Weight', value=1e-2)
st.write('Selected Style Weight is: {}'.format(style_weight))
num_iterations = st.slider('Number of iterations for training?', 0, 5000, 100)

row_col = st.number_input('Please Select Number of Images you want to plot', value=10)


# =============================================================================
# Compute Style Loss
# =============================================================================

def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  """Expects two images of dimension h, w, c"""
  # height, width, num filters of each layer
  # We scale the loss at a given layer by the size of the feature map and the number of filters
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)



def get_feature_representations(model, content_path, style_path):

  # Load our images in 
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)
  # print('style_output shape {}'.format(style_output.shape))

  
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features


# =============================================================================
# Compute Combine (style and content) loss
# =============================================================================
def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):

  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score 
  return loss, style_score, content_score



def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image

  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss


def run_style_transfer(content_path, 
                       style_path,
                       num_iterations,
                       content_weight, 
                       style_weight): 
    
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  image_placeholder = st.empty()
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = load_and_process_img(content_path)
  init_image = tf.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

  # For displaying intermediate images 
  iter_count = 1
  
  # Store our best result
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
    
  # For displaying
# =============================================================================
#   num_rows = 2
#   num_cols = 5
#   # display_interval =int(num_iterations/(num_rows*num_cols))
# =============================================================================
  display_interval = int(num_iterations/row_col)
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   

  st.sidebar.text("Training Progress")
  myBar = st.sidebar.progress(0)
  time_placeholder = st.sidebar.empty()
  imgs = []
  style_loss = []
  content_loss = []
  loss_title = st.empty()
  loss_placeholder = st.empty()
  
  for i in range(num_iterations):
    epochPerc = ((i+1)*100)/num_iterations
    myBar.progress(int(epochPerc))
    start_time = time.time()
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    style_loss.append(style_score.numpy())
    content_loss.append(content_score.numpy())
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    
    chart_data = pd.DataFrame()
    chart_data['Style_Loss'] = style_loss
    chart_data['Content_Loss'] = content_loss
    loss_title.markdown(" &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;The Below Interactive Plot Shows Content Loss / Style Loss vs Iteration")
    loss_placeholder.line_chart(chart_data, width=5)
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = deprocess_img(init_image.numpy())


    if i % display_interval == 0:
      # Use the .numpy() method to get the concrete numpy array
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      imgs.append(plot_img)      
      image_placeholder.image(plot_img, caption='iteration: {}'.format(i+1))

    time_placeholder.markdown('Estimated Time Left {}'.format(str(datetime.timedelta(seconds=(time.time() - start_time)*(num_iterations - i)))[:-7]))
    
  st.write('Total time: ',str(datetime.timedelta(seconds=(time.time() - global_start)))[:-7])
  st.write("The **Best Image** generated after the training is below. (Best Image: Image which has minimum **Loss** value.")
  st.image(best_img, caption="Best Image!")
  im222 = Image.fromarray(best_img)
  im222.save('best-image.jpg')
  time_placeholder.markdown('Total training time: {}'.format(str(datetime.timedelta(seconds=(time.time() - global_start)))[:-7]))

  return best_img, best_loss, imgs


## Training

if st.button('Start Training'):
    best, best_loss, imgs = run_style_transfer(content_path, style_path, num_iterations=num_iterations, content_weight=content_weight, style_weight=style_weight)

# =============================================================================
# Data Deletion and Celebration
# =============================================================================
if "Your-Content-Image.jpg" in ImageNames123 or "Your-Style-Image.jpg" in ImageNames123:
    vv = st.button("Delete Data")
    if vv:
        if "Your-Style-Image.jpg" in ImageNames123:
            os.remove("Your-Style-Image.jpg")
            style_sidebar.image(Image.open("Waves.jpg"), caption="Content Image", width=200)
        if "Your-Content-Image.jpg" in ImageNames123:
            os.remove("Your-Content-Image.jpg")
            content_sidebar.image(Image.open("Walhalla-Regensburg.jpg"), caption="Content Image", width=200)
        if "best-image.jpg" in ImageNames123:
            os.remove("best-image.jpg")

st.markdown("## Party time!")
st.write("Yay! You're done with this Training and Generation of NST image. Click below to celebrate.")
btn = st.button("Celebrate!")
if btn:
    st.balloons()
    
'''
#### Please feel free to write me any suggestions you have
You can reach me via [**LinkedIn**](https://www.linkedin.com/in/vedant-parikh-04923a41/) or [**E-mail**](mailto:vedant.parikh@outlook.com)
and feel free to have look at other Machine Learning / Data Science projects on my [**Github**](https://github.com/vedantparikh) page.
'''