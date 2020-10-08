# Sonnet Generation using an Image

### GPU USED IS "GeForce GTX 1080 Ti with Compute Capability 6.1"

### libraries tensorflow and numba for connecting with GPU

### large download ahead. You'll use the training set, which is a 13GB file MS-COCO dataset https://cocodataset.org/#home.

Researches that involve both vision and languages have attracted great attentions recently as we can witness from the bursting works on image descriptions like image caption and paragraph.Image descriptions aim to generate sentence(s) to describe facts from images in human-level languages. 


Given an image like the example below, our goal is to generate a caption such as "a delicious chocolate cake slice".

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/COCO_train2014_000000548913.jpg" width="590px" height="400px">

## For the Result, the project is divided into two parts i.e Image Captioning + Sonnet generation from Image Caption using LSTM RNN in Python with Keras 

Image captioning code https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Run.py  Run this code for image caption generation.

To accomplish this, I have used an attention-based model, which enables us to see what parts of the image the model focuses on as it generates a caption.

We have downloaded the MS-COCO dataset https://cocodataset.org/#home , preprocessed it and it caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

## Image Caption generation Explanation of code run

Download and prepare the MS-COCO dataset

The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. The code below downloads and extracts the dataset automatically.



