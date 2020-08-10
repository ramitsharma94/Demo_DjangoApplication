from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import joblib
import numpy as np
import pymongo
from pymongo import MongoClient
# Create your views here.

client = MongoClient("localhost", 27017)
db = client["mpgDataBase"]
collectionD = db["mpgTable"]
reloadModel = joblib.load("./model/RFModelForMPG.pkl")

def index(request):
    temp = {}
    temp["cylinders"] = 8
    temp["displacement"] = 307
    temp["horsepower"] = 130
    temp["weight"] = 3504
    temp["acceleration"] = 12
    temp["model_year"] = 70
    temp["origin"] = 1

    context = {"temp": temp}
    return render(request, "index.html", context)



def predictMPG(request):
    if(request.method == "POST"):
        print(request.POST.get("originVal"))
        temp = {}
        temp["cylinders"] = request.POST.get("cylinderVal")
        temp["displacement"] = request.POST.get("dispVal")
        temp["horsepower"] = request.POST.get("hrsPwrVal")
        temp["weight"] = request.POST.get("weightVal")
        temp["acceleration"] = request.POST.get("accVal")
        temp["model year"] = request.POST.get("modelVal")
        temp["origin"] = request.POST.get("originVal")


    testData = pd.DataFrame({"x": temp}).transpose()
    testData["origin"] = testData["origin"].astype(np.uint)
    print(testData.head())
    scoreVal = reloadModel.predict(testData)[0]
    context = {"scoreVal": scoreVal, "temp": temp}
    return render(request, "index.html", context)

def viewDatabase(request):
    countofrow = collectionD.find().count()
    context = {"countofrow": countofrow}
    return render(request, "viewDB.html", context)

def updateDatabase(request):
    temp = {}
    temp["cylinders"] = request.POST.get("cylinderVal")
    temp["displacement"] = request.POST.get("dispVal")
    temp["horsepower"] = request.POST.get("hrsPwrVal")
    temp["weight"] = request.POST.get("weightVal")
    temp["acceleration"] = request.POST.get("accVal")
    temp["model year"] = request.POST.get("modelVal")
    temp["origin"] = request.POST.get("originVal")
    temp["mpg"] = request.POST.get("mpgVal")
    collectionD.insert_one(temp)

    countofrow = collectionD.find().count()
    context = {"countofrow": countofrow}
    return render(request, "viewDB.html", context)