# importing
import model_processing
import connection_handler
import json
import os
import cv2
import pathlib

# from _thread import *
# import threading
# print_lock = threading.Lock()

def connect_to(ipaddr, port):
    while True:
        try:
            listener = connection_handler.TCPConnectionHandler(ipaddr, port)
            listener.setup_server()
            listener.listen()
        except Exception as e: 
            print("Cannot connect to client")
            print("Error: {}".format(e))
            return None
        else:
            return listener

def get_model_handler(model_name):
    while True:
        try:
            print("Model: {}.".format(model_name))
            model_handler = model_processing.Model(model_name)
            model_handler.get_model().summary()
        except: 
            print("Cannot create model")
            return None
        else:
            print("Created model")
            return model_handler

def socket_sending_process(listener, context, is_sending_str=True):
    # Sending message length

    while listener.recv():
        print("Received pipeline")
        if is_sending_str:
            listener.send(context)
            listener.send("Done")
        else:                          
            listener.send(str(len(context)))
            print("Sent length: " + str(len(context)))

            while listener.recv():
                print("Sending file . . .")
                listener.send_bytes(context)
                break
        print("Sent pipeline")
        return

def prepare_to_send_model_name(listener):
    if listener.recv():
        listener.send("Done")
        if listener.get_bytes_received_str() == "Sending":
            return True

def get_client_message(listener):
    while listener.recv():
        listener.send("Done")
        return listener.get_bytes_received_str()

def get_model_name(listener):
    return get_client_message(listener)

def imagefile_to_bytes(image_file):
    # Input image
    file = os.path.join(os.path.dirname(os.path.realpath(__file__)), image_file)
    print(file)
    image = cv2.imread(file)

    print(image.shape)
    im_resize = cv2.resize(image, (32, 32))

    byte_image = cv2.imencode('.jpg', im_resize)[1].tobytes()
    print(len(byte_image))

    return byte_image

def main() -> None:
    """Demo
    Connect to a client socket to send and receive pipeline data

    Serialize a Model and send it to the client socket as json

    Read incoming photo

    Send output photos as bytes
    """

    # 
    # Connection
    #

    # url_client = "tcp://*:54321"

    while True:
        listener = None
        model_handler = None
        PORT = 54321
        IP_ADDR = "0.0.0.0"

        listener = connect_to(IP_ADDR, PORT)
        if listener == None:
            continue
            
        # 
        # Model
        #

        while prepare_to_send_model_name(listener):
            # model_name = "AlexNetModel"
            # model_handler = get_model_handler()
            model_name = get_model_name(listener)
            model_handler = get_model_handler(model_name)
            if model_handler == None:
                break

            model_json = model_handler.get_model_json()


            socket_sending_process(listener, model_json)

            print(model_json)

            
            # 
            # Output shape
            #

            model_output_shape_json = model_handler.get_output_shape()

            socket_sending_process(listener, model_output_shape_json)
            print(model_output_shape_json)

            
            # 
            # Input photo
            #

            file = "Photo/sadtest.jpg"

            # image_bytes = imagefile_to_bytes(file)
            # socket_sending_process(listener, image_bytes, is_sending_str=False)

            # 
            # Output photo
            # #
            list_of_image_bytes = model_handler.get_list_of_image_bytes(file)

            count = 0

            for image_bytes_in_a_layer in list_of_image_bytes:
                for image_bytes in image_bytes_in_a_layer:
                    print("Sending . . .")
                    socket_sending_process(listener, image_bytes, is_sending_str=False)
                    count += 1
                    print(count)

            print("Transfer model {} infomation to client completely".format(model_name))


if __name__ == '__main__':
    main()