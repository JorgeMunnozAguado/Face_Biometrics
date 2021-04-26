
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Activation, ActivityRegularization
from keras_vggface.vggface import VGGFace

from embeddings import loadImage
from classifier import evaluate, defineModel, loadModel, completeModel


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)

    # plt.imshow(img)
    # plt.show()

    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        print('--------->>', preds)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

        print(class_channel)

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    print('---------')
    print(heatmap)
    print('---------')
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))



def model_load(title, verbose=0):

    # Backbone + classification layers
    model_base = completeModel(2)

    # Define auxiliar model
    model_simple = defineModel((None, 2048),2)
    model_simple = loadModel(model_simple, title + '_checkpoint_model')

    # Load weights (from small model)
    model_base.layers[-1].set_weights(model_simple.layers[-1].get_weights())
    model_base.layers[-2].set_weights(model_simple.layers[-2].get_weights())
    model_base.layers[-3].set_weights(model_simple.layers[-3].get_weights())

    if verbose: model_base.summary()


    return model_base



def image_load(path):
    pass



if __name__ == '__main__':

    # Prepare image
    # img_array = preprocess_input(get_img_array(img_path, size=img_size))

    title = 'Asiatico'
    image_size = 224
    image_size_t = (224, 224)
    path = '4K_120/HA4K_120/10011748@N08_identity_0'

    preprocess_input = keras.applications.xception.preprocess_input


    img, img_path = loadImage(path, image_size)
    img_array = get_img_array(img_path, image_size_t)
    img_array = preprocess_input(img_array)


    model_base = model_load(title)


    # Generate class activation heatmap
    # heatmap = make_gradcam_heatmap(img_array, model_base, 'avg_pool')
    heatmap = make_gradcam_heatmap(img_array, model_base, 'conv5_3_1x1_increase/bn')


    save_and_display_gradcam(img_path, heatmap)

    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # TODO - save heatmap

    save_and_display_gradcam(img_path, heatmap)
