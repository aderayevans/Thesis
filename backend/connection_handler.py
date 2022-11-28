import socket
import numpy as np

class TCPConnectionHandler:

    def __init__(self, IP_ADDR, PORT):
        self.ip_address = IP_ADDR
        self.port = PORT

    def setup_server(self):
        # create a socket
        self.__create_socket()
        # bind the socket to a ip / port
        self.__bind()
        # mark the socket for listening
        self.__listen()

    def __create_socket(self):
        self.listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def __bind(self):
        self.listener_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener_socket.bind((self.ip_address, self.port))
    def __listen(self):
        self.listener_socket.listen(10)
        # self.listener_socket.send_string(f"Hello from ZMQ server")

    def listen(self):
        print("listening on {}:{}".format(self.ip_address, self.port))
        self.client_socket, self.addr = self.listener_socket.accept()
        print("[ACCEPTED] at {}:{}".format(self.addr, self.port))
        # while True:
        #     self.bytes_received = self.listener_socket.recv(1024)
        #     self.array_received = np.frombuffer(self.bytes_received, dtype=np.float32).reshape(28, 28)

    def get_bytes_received_str(self):
        return self.bytes_received.decode('utf-8')[:-2] #\r\n

    def recv(self):
        while True:
            self.bytes_received = self.client_socket.recv(1024)
            print("Received: {}".format(self.bytes_received.decode('utf-8')))
            return True

    def send_back(self):
        self.send_bytes(self.bytes_received)

    def send_bytes(self, bytes_to_send):
        # pred = model.predict(self.array_received.reshape(1, 784))
        # self.bytes_to_send = pred.tobytes()
        self.client_socket.send(bytes_to_send)

    def send(self, string_to_send):
        bytes_to_send = string_to_send.encode('utf-8')
        self.send_bytes(bytes_to_send)