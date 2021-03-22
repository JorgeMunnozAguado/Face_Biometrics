
from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Activation, ActivityRegularization

#Import the ResNet-50 model trained with VGG2 database
my_model = 'resnet50'
resnet = VGGFace(model = my_model)
#resnet.summary()

#Select the lat leayer as feature embedding
last_layer = resnet.get_layer('avg_pool').output
feature_layer = Flatten(name='flatten')(last_layer)
model_vgg=Model(resnet.input, feature_layer)
