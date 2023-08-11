import cv2
import numpy as np
from fastapi import FastAPI, Request
import pickle
import base64
from app.hog import gethog
# Load the image as grayscale

# import cv2
# import numpy as np
# # Load the image as grayscale
# img_gray = cv2.imread('C:\\2178\\Cars Dataset\\test\Audi\\42.jpg', 0)
# win_size = img_gray.shape
# cell_size = (8, 8)
# block_size = (16, 16)
# block_stride = (8, 8)
# num_bins = 9
# # Set the parameters of the HOG descriptor using the variablesdefined above
# hog = cv2.HOGDescriptor(win_size, block_size, block_stride,cell_size, num_bins)
# # Compute the HOG Descriptor for the gray scale image
# hog_descriptor = hog.compute(img_gray)
# print ('HOG Descriptor:', hog_descriptor)

app = FastAPI()

def readb64(url):
    encoded_data = url.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data),np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


@app.get("/api/gethog")
async def read_str(request: Request):
        item  =   await request.json()
        item_str = item['img']
        img = readb64(item_str)
        hog = gethog(img)
        return {"hog":hog.tolist()}
