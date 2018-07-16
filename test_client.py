import xmlrpc.client
import numpy as np

with xmlrpc.client.ServerProxy("http://localhost:8000/") as proxy:
#    print("3 is even: %s" % str(proxy.is_even(3)))
    data = proxy.convert('banner-car.png')


[print(line) for line in data]