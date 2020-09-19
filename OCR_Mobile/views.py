from django.shortcuts import render
from django.http import HttpResponse
from CRAFT import ServerRun
from ocr_amit import run_ocr
import os
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import shutil

net = ServerRun.createModel()
result_folder = './result/'
if not os.path.isdir(result_folder):
            os.mkdir(result_folder)

ocr_engines = run_ocr.getReadTextModel()


def simple_upload(request):
    if request.method == 'POST' and request.FILES['myfile']:
        result = 'We received your file'
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)[1:]
        shutil.rmtree('./result')
        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
        uploaded_file_url = os.path.join(settings.BASE_DIR,uploaded_file_url)
        bboxes, polys, score_text = ServerRun.runLineCut(
            uploaded_file_url, net, './result')
        result = run_ocr.runReadText('./result', ocr_engines)
        return HttpResponse(result)
        
    return render(request, 'OCR_Mobile/core/simple_upload.html')

def checking(request):
    return HttpResponse('Ok')



# Create your views here.
