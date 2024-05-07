# Prompt-CLIP-3V3D
The code for 'Prompt CLIP-3V3D: Multiple Prompt-Driven 3D Medical Image Segmentation Model'

The test results on the ADNI1 dataset can be obtained using the test_adni.py file.

Start the testing process in just three steps.

First, download the ADNI1 dataset from Google Drive(https://drive.google.com/drive/folders/1cu-XL9Ju5YTn6N6OTuhSctNoBLbIeA6c?usp=sharing). Unzip the compressed file to obtain the dataset, then place the dataset into the dataset directory while maintaining the directory structure dataset/ADNI/ADNI1-M-TRAIN. 

Similarly, place the other files in the same manner as the ADNI1-M-TRAIN.

Second, download the weight file: pretrained_weights.zip, Generator40000.pth.

Unzip the pretrained_weights.zip file and place the files inside the pretrained_weights/ directory of this project. 

Place the Generator40000.pth inside the snapshots-adni/ directory of this project.

Third, run test_adni.py


