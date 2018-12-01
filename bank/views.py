from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# Create your views here.
def login(request):
    return render(request, 'bank/login.html', locals())