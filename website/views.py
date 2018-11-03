from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse

def index(request):
    title = 'main'
    return render(request, 'website/index.html', locals())

def camera(request):
    title = 'camera'
    return render(request, 'website/camera_test.html', locals())

def uploadImage(request):
    if request.method == 'POST':
        print('uploadImage start...')
        imgString = request.POST['image']
        imgString = imgString.replace('data:image/png;base64,', '')

        import base64
        imgdata = base64.b64decode(imgString)
        filename = 'cameraCapture.png'  
        with open(filename, 'wb') as f:
            f.write(imgdata)
        print('uploadImage end...')
        return HttpResponse('uploadImage done')
