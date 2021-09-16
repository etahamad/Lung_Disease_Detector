import os
from django.shortcuts import render
from django.core.files.storage import default_storage
from .LungD import Lung_Disease_Detector


# Create your views here.
def Home(request):
    return render(request, 'index.html')


def PredictionPage(request):
    if request.method == 'POST':
        try:
            image = request.FILES['image']
            image_name = default_storage.save(image.name, image)
            image_url = default_storage.path(image_name)

            classifier = Lung_Disease_Detector()
            case = classifier.Classify(img_path=image_url)

            context = {
                'pred': "Your Case is " + case[0],
                'image_url': '/media/' + image_name,
                'treatment': 'Treatment : ' + case[1]
            }

        except Exception as e:
            context = {
                'pred': " ",
                'image_url': " ",
                'treatment': " "
            }
            print(e)

        return render(request, 'prediction.html', context)
    return render(request, 'prediction.html')


