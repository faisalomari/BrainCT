import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image  # Import PIL library for image processing
import shutil
from shutil import copy_image

currentDir = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0')
datasetDir = str(Path(currentDir, 'Patients_CT'))

# Reading labels
hemorrhage_diagnosis_df = pd.read_csv(
    Path(currentDir, 'hemorrhage_diagnosis.csv')
)
hemorrhage_diagnosis_array = hemorrhage_diagnosis_df.values  # Use .values instead of ._get_values
print(hemorrhage_diagnosis_array[0][0])


counter = np.zeros((2, 7))
# 0 - with
# 1 - without

t1 = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/data_splitted_bone')
t2 = Path('/home/faisal/Desktop/BrainDL/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/data_splitted_brain')
for i in range(2501):
    patient = hemorrhage_diagnosis_array[i][0]
    slice = hemorrhage_diagnosis_array[i][1]
    if patient<100:
        patient = '0' + str(patient)
    else:
        patient = str(patient)
    s = str(Path(datasetDir, patient))
    s1 = str(Path(s, 'bone/' + str(slice) + '.jpg'))
    s2 = str(Path(s, 'brain/' + str(slice) + '.jpg'))
    print(s1)
    print(s2)
    if(hemorrhage_diagnosis_array[i][2] == 1):
        t = str(t1) + '/Intraventricular/with/' + str(int(counter[0][0])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Intraventricular/with/' + str(int(counter[0][0])) + '.jpg'
        copy_image(s2,t)
        counter[0][0] +=1
    else:
        t = str(t1) + '/Intraventricular/without/' + str(int(counter[1][0])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Intraventricular/without/' + str(int(counter[1][0])) + '.jpg'
        copy_image(s2,t)
        counter[1][0] +=1

    if(hemorrhage_diagnosis_array[i][3] == 1):
        t = str(t1) + '/Intraparenchymal/with/' + str(int(counter[0][1])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Intraparenchymal/with/' + str(int(counter[0][1])) + '.jpg'
        copy_image(s2,t)
        counter[0][1] +=1
    else:
        t = str(t1) + '/Intraparenchymal/without/' + str(int(counter[1][1])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Intraparenchymal/without/' + str(int(counter[1][1])) + '.jpg'
        copy_image(s2,t)
        counter[1][1] +=1

    if(hemorrhage_diagnosis_array[i][4] == 1):
        t = str(t1) + '/Subarachnoid/with/' + str(int(counter[0][2])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Subarachnoid/with/' + str(int(counter[0][2])) + '.jpg'
        copy_image(s2,t)
        counter[0][2] +=1
    else:
        t = str(t1) + '/Subarachnoid/without/' + str(int(counter[1][1])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Subarachnoid/without/' + str(int(counter[1][1])) + '.jpg'
        copy_image(s2,t)
        counter[1][2] +=1

    if(hemorrhage_diagnosis_array[i][5] == 1):
        t = str(t1) + '/Epidural/with/' + str(int(counter[0][3])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Epidural/with/' + str(int(counter[0][3])) + '.jpg'
        copy_image(s2,t)
        counter[0][3] +=1
    else:
        t = str(t1) + '/Epidural/without/' + str(int(counter[1][3])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Epidural/without/' + str(int(counter[1][3])) + '.jpg'
        copy_image(s2,t)
        counter[1][3] +=1

    if(hemorrhage_diagnosis_array[i][6] == 1):
        t = str(t1) + '/Subdural/with/' + str(int(counter[0][4])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Subdural/with/' + str(int(counter[0][4])) + '.jpg'
        copy_image(s2,t)
        counter[0][4] +=1
    else:
        t = str(t1) + '/Subdural/without/' + str(int(counter[1][4])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Subdural/without/' + str(int(counter[1][4])) + '.jpg'
        copy_image(s2,t)
        counter[1][4] +=1

    if(hemorrhage_diagnosis_array[i][7] == 1):
        t = str(t1) + '/No_Hemorrhage/with/' + str(int(counter[0][5])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/No_Hemorrhage/with/' + str(int(counter[0][5])) + '.jpg'
        copy_image(s2,t)
        counter[0][5] +=1
    else:
        t = str(t1) + '/No_Hemorrhage/without/' + str(int(counter[1][5])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/No_Hemorrhage/without/' + str(int(counter[1][5])) + '.jpg'
        copy_image(s2,t)
        counter[1][5] +=1

    if(hemorrhage_diagnosis_array[i][8] == 1):
        t = str(t1) + '/Fracture_Yes_No/with/' + str(int(counter[0][6])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Fracture_Yes_No/with/' + str(int(counter[0][6])) + '.jpg'
        copy_image(s2,t)
        counter[0][6] +=1
    else:
        t = str(t1) + '/Fracture_Yes_No/without/' + str(int(counter[1][6])) + '.jpg'
        copy_image(s1,t)
        t = str(t2) + '/Fracture_Yes_No/without/' + str(int(counter[1][6])) + '.jpg'
        copy_image(s2,t)
        counter[1][6] +=1