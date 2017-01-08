from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
from django.http import HttpResponseRedirect
from . import classify

def index(request):
    return HttpResponse("Hello, world.")


class homePage(View):
    template_name = "index.html"

    def get(self, request):
        return render(request, self.template_name)

class resultView(View,classify.classify):
    template_name = 'result.html'
    result = ""

    def get(self, request):

        if 'query' in request.GET and request.GET['query']:
            query = request.GET['query']
            classMe = classify.classify.feature_extraction_classify(query)
            return render(request, self.template_name, {'query': query, 'result': classMe})

        else:
            message_two = 'You submitted an empty form'
            self.message = message_two

        return render(request, self.template_name, {'query': self.message})

