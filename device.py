##% For test
import pandas as pd


#%% Image resizer
from PIL import Image
import numpy as np

def convert(image_path):
    """Converts an image into a grayscale uint8 np array"""
    x = Image.open(image_path).convert('LA')
    x = x.convert('L')
    y = np.asarray(x.getdata(),dtype=np.float64).reshape((x.size[1],x.size[0]))
    
    return y

data = convert("banner-car.png")


#%% Example RPCserver
from xmlrpc.server import SimpleXMLRPCServer

def is_even(n):
    return n % 2 == 0

server = SimpleXMLRPCServer(("localhost", 8000))
print("Listening on port 8000...")
server.register_function(is_even, "is_even")
server.serve_forever()

#%%
from PIL import Image
import numpy as np
from xmlrpc.server import SimpleXMLRPCServer



class DeviceController():
    
    def __init__(self):
        print('Initiated')

    def start(self):
        """Starts XML server"""
        server = SimpleXMLRPCServer(("localhost", 8000))
        print("Listening on port 8000...")
        server.register_function(self.convert, "convert")
        server.serve_forever()

    def get_image(self, image_path):
        x = Image.open(image_path).convert('LA').resize((28, 28))

        return x

    def convert(self, image_data):
        """Converts an image into a grayscale uint8 np array"""
        image_data = image_data.convert('L')
        y = np.asarray(image_data.getdata(),dtype=np.uint8).reshape((image_data.size[1],image_data.size[0]))
        
        return y.tolist()
    
    
#%%
dc = DeviceController()
data = dc.convert(dc.get_image('banner-car.png'))
#dc.start()

#%%




































