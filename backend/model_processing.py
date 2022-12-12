import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
IMAGE_SHAPE = (320, 320)
INPUT_SHAPE = (320, 320, 3)


class Model:
    def __init__(self, model_name=None):
        if model_name is None:
            self.model = AlexNetModel()
        elif model_name == "AlexNet":
            self.model = AlexNetModel()
        elif model_name == "LeNet5":
            self.model = LeNet5Model()
        elif model_name == "VGG16":
            self.model = VGG16Model()
        else:
            self.set_model(model_name)

        self.list_of_image_bytes = []
        self.count = 0
        # Display the model's architecture
        # self.model.summary()

    def set_model(self, model_name):
        # self.model = keras.Sequential([
        #     hub.KerasLayer(model_name)
        # ])
        # self.model.build((320, 320, 3))

        # IMAGE_SHAPE = (320, 320)
        # self.model = keras.Sequential([
        #     hub.KerasLayer(model_name, input_shape=IMAGE_SHAPE+(3,))
        # ])

        # self.model = tf.saved_model.load(model_name)
        self.model = tf.keras.models.load_model(model_name)

    def get_model(self):
        return self.model

    def get_layers(self):
        return self.model.layers

    def get_output_shape(self):
        _list = []
        for layer in self.get_layers():
            _list.append(layer.output_shape)

        _dict = {}
        _dict["content"] = _list

        return json.dumps(_dict)

    def get_model_json(self):
        return self.model.to_json()

    def set_input_image(self, image):
        image = cv2.resize(image, IMAGE_SHAPE, interpolation = cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image

    def processing_image_through_model(self):
        (B, G, R) = cv2.split(self.image)
        self.list_of_image_bytes.append([image_to_bytes(B), image_to_bytes(G), image_to_bytes(R)])
        self.count += 3
        
        # self.list_of_image_bytes.append(image_to_bytes(self.image))
        # self.count += 1

        tensor_input = np.array([cv2.resize(self.image, (self.model.input_shape[1], self.model.input_shape[2])), ], dtype=float) / 255.
        for layer in self.model.layers:
            image_bytes_in_a_layer = []

            tensor_input = layer(tensor_input)
            tensor_output = tensor_input
            tensor_output = tensor_output.numpy()

            print(layer.name)
            print(tensor_output.shape)
            print(self.count)

            if len(tensor_output.shape) == 4:
                number_of_images = tensor_output.shape[-1]

                for i in range(number_of_images):
                    tensor_image = 255 - (tensor_output[0, :, :, i] * 255)
                    tensor_image = cv2.cvtColor(tensor_image, cv2.COLOR_BGR2RGB)
                    image_bytes_in_a_layer.append(image_to_bytes(tensor_image))
                    self.count += 1


            self.list_of_image_bytes.append(image_bytes_in_a_layer)

    def get_output_images(self):
        return self._list

    def get_list_of_image_bytes(self, image_file_path):
        image_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), image_file_path)
        self.set_input_image(cv2.imread(image_file))

        self.processing_image_through_model()

        return self.list_of_image_bytes


def AlexNetModel():
    AlexNet = tf.keras.Sequential([
        #1st Convolutional Layer
        tf.keras.layers.Conv2D(filters=96, input_shape=INPUT_SHAPE, kernel_size=(11,11), strides=(4,4), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

        #2nd Convolutional Layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

        #3rd Convolutional Layer
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        #4th Convolutional Layer
        tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        #5th Convolutional Layer
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

        #Passing it to a Fully Connected layer
        tf.keras.layers.Flatten(),
        # 1st Fully Connected Layer
        tf.keras.layers.Dense(4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        # Add Dropout to prevent overfitting
        tf.keras.layers.Dropout(0.4),

        #2nd Fully Connected Layer
        tf.keras.layers.Dense(4096),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        #Add Dropout
        tf.keras.layers.Dropout(0.4),

        #3rd Fully Connected Layer
        tf.keras.layers.Dense(1000),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        #Add Dropout
        tf.keras.layers.Dropout(0.4),

        #Output Layer
        tf.keras.layers.Dense(10),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('softmax'),

    ])

    return AlexNet

def LeNet5Model():
    LeNet5 = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.AveragePooling2D(),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    return LeNet5

def VGG16Model():
    model = tf.keras.applications.vgg16.VGG16()
    model2 = tf.keras.Model(model.input, model.layers[-2].output)
    return model2

def image_to_bytes(image):
    # print(image.shape)
    # image = cv2.resize(image, (32, 32))

    byte_image = cv2.imencode('.png', image)[1].tobytes()
    # print(len(byte_image))

    return byte_image

def change_input_shape(model):
    new_input_shape = (None, 250, 250, 3)

    model.layers[0].batch_input_shape = new_input_shape

    # rebuild model architecture by exporting and importing via json
    new_model = tf.keras.models.model_from_json(model.to_json())
    new_model.summary()

    # copy weights from old model to new one
    # for layer, new_layer in zip(model.layers,new_model.layers):
    #     new_layer.set_weights(layer.get_weights())

    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    
    new_model.summary()

    return new_model

def demo():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models/imagenet_mobilenet_v2_100_128_classification_3_default_1/model.json")

    model = tf.keras.models.model_from_json(open(path).read())

    model.summary()
    model.layers

def main():

    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../models/saved_model/my_model")
    # path = "VGG16"
    path = "AlexNet"
    model_handler = Model(path)
    model = model_handler.get_model()
    model.summary()

    # model = change_input_shape(model)

    # Input image
    image_file = "Photo/sadtest.jpg"
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), image_file)
    image = cv2.imread(file)
    image = cv2.resize(image, IMAGE_SHAPE, interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor_input = np.array([cv2.resize(image, IMAGE_SHAPE), ], dtype=float) / 255.
    for layer in model.layers:
        tensor_input = layer(tensor_input)
        tensor_output = tensor_input
        tensor_output = tensor_output.numpy()

        print(layer.name)
        print(tensor_output.shape)

        if len(tensor_output.shape) == 4:
            number_of_images = tensor_output.shape[-1]
            edge = int(np.ceil(np.sqrt(number_of_images)))

            print(number_of_images)
            print("edge: ", edge)

            for i in range(number_of_images):
                tensor_image = 255 - ((tensor_output[0, :, :, i] * 255))
                # tensor_image = cv2.bitwise_not(tensor_image)
                plt.subplot(edge, edge, i+1)
                tensor_image = cv2.cvtColor(tensor_image, cv2.COLOR_BGR2RGB)
                plt.imshow(tensor_image.astype('uint8'))
            plt.show()

if __name__ == '__main__':
    main()

    # demo()