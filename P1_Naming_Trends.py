# Python Project - 1
# Project: Analyzing the naming trends using Python 
# Code for finding Gender Distribution and Popular baby names from 1880 to 2018 National data for USA.
# For running code, keep nmes.Zip file in same folder and run as python P1_Naming_Trends.py
import sys 
import pandas as pd
import zipfile
import os
from os import listdir
from collections import defaultdict
import matplotlib.pyplot as plt

def gender_counts(file):
    gender = {}
    gender_counts = {}
    txt = []
    yob = []
    if '.txt' in file:
        with open(file) as file_open:
            for line in file_open:
                row = line.strip().split(",")
                status = str(row[1])
                count = int(row[2])
                if status in gender:
                    value = gender[status]
                    gender[status] = value + count
                else:
                    gender[status] = count

        for status in gender:
            txt = file.strip('yob')
            yob = txt.strip('.txt')
            print(yob,status,gender[status])

    else:
        print(file,' IS NOT USEFUL FILE\n')
    
baby_names = {}     
def popular_names(file):
    if '.txt' in file:
        with open(file) as file_open:
            for line in file_open:
                row = line.strip().split(",")
                names = str(row[0])
                count = int(row[2])
                if names in baby_names:
                    value = baby_names[names]
                    baby_names[names] = value + count
                else:
                    baby_names[names] = count

    else:
        print(file,' IS NOT USEFUL FILE\n')
        
def top_names(baby_names,n):
    dct = defaultdict(list)
    for k, v in baby_names.items():
        dct[v].append(k)
    baby_names = dict(sorted(dct.items())[-n:][::-1])
    for names in baby_names:
        print(baby_names[names]," was named ",names," times.")
    
def main():
    # Reading Zip File and Extracting it into a folder
    names_folder = zipfile.ZipFile("names.zip", mode='r', compression=zipfile.ZIP_DEFLATED)
    print("\nExtracting....")
    names_folder.extractall()
    print("\nExtracting Completed.\n")

    # Getting file_names from the extracted folder.
    print("Analyzing Data from files....\n")
    from os.path import isfile, join
    file_names = [f for f in listdir("names_extracted/") if isfile(join("names_extracted/", f))]
    
    # Year-wise Gender Distribution.    
    print("Yearwise Gender Distribution.\n")
    print("YOB", "Sex", "Count")
    for file in file_names:
        gender_counts(file)
        popular_names(file)
        
    print("\nPropular Baby Names. (Top 100)\n")
    top_names(baby_names,100)
    
    names_folder.close()

if __name__ == '__main__':
    sys.exit(main())