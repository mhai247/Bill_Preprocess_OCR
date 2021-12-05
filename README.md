# Text Image Scan
## Introduciton
This repo is to scan and align a text image file using text detection.
Like this
![Result](result.jpg "Result")
## Pipeline
1. Image rotation using OpenCV
1. Text detection using Paddle OCR
1. Maping detected box in a new image using pyimagesearch
1. OCR
## Use

Clone this repo

```
git clone https://github.com/mhai247/Text_Image_Scan
cd Bill_Preprocess_OCR
```
Install require package

```
pip install -r requirement.txt
```

Test the avaiable dataset

```
python3 scan.py
```

Copy your dataset into data folder
Run in your dataset

```
python3 scan.py --image_dir YOUR_DATASET_NAME
```

## Opensource code
[pyimagesearch](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[vietocr](https://github.com/pbcquoc/vietocr)