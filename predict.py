import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model

model_path = r'D:\Models\Braintumor'
# print(os.listdir(model_path))
model = load_model(os.path.join(model_path,'ICmodel_1.h5'))

print("Choose file to classify BrainTumor:")
filename = askopenfilename(initialdir=r'D:\BrainTumorDetector\Testing',title='Select Image:')

img = cv2.imread(filename)
print("Image chosen:")
fig,axs = plt.subplots()
axs.imshow(img)

img = cv2.resize(img,(256,256))
print(img.shape)
pred = model.predict(np.expand_dims(img,axis=0))

if np.argmax(pred)==0:
    print("Glioma")
elif np.argmax(pred)==1:
    print("Meningioma")
elif np.argmax(pred)==2:
    print("No Tumor")
elif np.argmax(pred)==3:
    print("Pituitary")
