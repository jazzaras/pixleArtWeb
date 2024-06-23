import base64
import cv2
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from django.template import RequestContext
from django.views.decorators.csrf import csrf_protect
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image 
from io import BytesIO



def home(request):
    if(request.POST):
        print("in")
        return conv(request)
          
    return render(request, "pixle/home.html")


@csrf_protect
def conv(request):
    csrfContext = RequestContext(request)

    fileUpload = request.FILES['fileToUpload'].read()
    pixleSize = int(request.POST['pixleSize'])
    nColors = int(request.POST['nColors'])
    image = cv2.imdecode(np.frombuffer(fileUpload , np.uint8), cv2.IMREAD_UNCHANGED)
    pixlisedImage = convertToPixles(image, pixleSize, nColors)
    # pixlisedImage = base64.b64encode(pixlisedImage)
    # pixlisedImage = base64.decodestring(pixlisedImage)
    print("ddddddd")
    pil_image = to_image(pixlisedImage)
    image_uri = to_data_uri(pil_image)
    context = {
        "image_uri":image_uri
    }
    return render(request, "pixle/home.html", context)




def convertToPixles(image, kernal_size = 10, n_clusters=5, ):
    h, w = image.shape[:2]

    clt = KMeans(n_clusters=n_clusters)
    clt.fit(image.reshape(-1, 3))

    clt_1 = clt.fit(image.reshape(-1, 3))

    # colorGrouping1 = 20
    # colorGrouping2 = 12
    # colorGrouping3 = 30

    i = 0
    j = 0

    counter = 0
    while (i + kernal_size < image.shape[0]):
        while (j + kernal_size < image.shape[1]):
            block = image[i:i+kernal_size, j:j+kernal_size]
            sum =  np.array([0, 0, 0])
            
            # know the block var have 100 pixels 
            # rows
            for k in block: 
                # columns
                for r in k: 
                    sum += r
            # print(block.shape)
            leng = (kernal_size*kernal_size)
            out = np.array([sum[0]/leng, sum[1]/leng, sum[2]/leng])

            nearestColor = clt.cluster_centers_[0]

            for color in clt.cluster_centers_:
                distOfCurr = np.linalg.norm(out-nearestColor)
                distOfNext = np.linalg.norm(out-color)

                if distOfNext < distOfCurr:
                    nearestColor = color

            out = nearestColor

            # rounding 
            # out = [round_to_multi(out[0], colorGrouping1), round_to_multi(out[1], colorGrouping2), round_to_multi(out[2], colorGrouping3)]


            image[i:i+kernal_size, j:j+kernal_size] = out
            
            # out = cv2.copyMakeBorder(
            #     image[i:i+kernal_size-2, j:j+kernal_size-2], 
            #     1, 
            #     1, 
            #     1, 
            #     1, 
            #     cv2.BORDER_CONSTANT, 
            #     value=[0,0,0]
            # )

            image[i:i+kernal_size, j:j+kernal_size] = out

            j += kernal_size
            
        i += kernal_size
        j = 0


    # print("ssss")
    # print(test[0])
    # print(test[0])
    # print(test[0])


    # if (h > 1000 ):
        # image = cv2.resize(image, (600, int((h * 600)/w)))



    # for i in test:
    # cv2.imshow("ima", image)

    # cv2.waitKey(0)


    return image


def round_to_multi(number, multiple):
    return multiple * round(number / multiple)

def show_img_compar(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

def to_image(numpy_img):
    numpy_img = numpy_img[:, :, ::-1] # this line changes the channels postion rbg --> bgr
    img = Image.fromarray(numpy_img, 'RGB')
    return img

def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "JPEG") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8')

def palette(clusters):
    width=300
    palette = np.zeros((50, width, 3), np.uint8)
    steps = width/clusters.cluster_centers_.shape[0]
    for idx, centers in enumerate(clusters.cluster_centers_): 
        palette[:, int(idx*steps):(int((idx+1)*steps)), :] = centers
    return palette
