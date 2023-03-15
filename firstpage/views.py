import cv2
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Create your views here.
def index(request):
    return render(request,'index.html')

def predict(request):
    fileObj = request.FILES['file_img']

    fs = FileSystemStorage() # object to save the image
    filename = fs.save(fileObj.name,fileObj) #function to save the image
    filename = fs.url(filename)
    img_path = "."+filename
    
    img = cv2.imread(img_path)
    #  if(png):
    #             x = x[...,:3]
    img=img[...,:3]
    img = cv2.resize(img,(128,128))
    x = np.array(img)
    generator=load_model('models\generator.h5')
    x = np.reshape(x,(1,128,128,3))
    gen_imgs = generator.predict(x)
    a = np.reshape(gen_imgs,(256,256,3))
    a = image.array_to_img(a)
    no = np.random.normal(0,1, (1, 1))
    no = no[0][0]
    a.save('./media'+"/"+str(no)+".jpg")
    img = './media'+"/"+str(no)+".jpg"
    context = {'filename':filename,'img':img}
    return render(request,'predict.html',context)