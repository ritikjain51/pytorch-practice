import streamlit as st
from model import ClassificationModel
from config import MODEL_PATH
import numpy as np
import torch
from PIL import Image

@st.cache
def load_model():
    model = torch.load(MODEL_PATH)
    return model

model = load_model()
model = model.to("cpu")
st.title("Image Classification - CIFAR 100")

st.markdown("The application is backup with pytorch for image classification")


img_file = st.file_uploader("Browse the image for classification", type = ["png", "jpeg", "jpg"])
if img_file:
    import pdb; pdb.set_trace()
    st.write(img_file)
    st.image(img_file)  
    im = Image.open(img_file)

    im_arr = np.array(im)
    st.write(im_arr.shape)
    im_arr = np.moveaxis(im_arr, -1, 0)
    im_arr = np.expand_dims(im_arr, axis = 0) 
    st.write(im_arr.shape)
    im_ten = torch.tensor(im_arr, dtype=torch.float64)
    st.write("Tensor dtype: ", im_ten.dtype)
    st.write("Tensor size: ", im_ten.size())
    resp = model(im_ten) 
    st.write(resp)