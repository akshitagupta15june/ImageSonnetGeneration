# Sonnet Generation using Image Captioning

### GPU USED IS "GeForce GTX 1080 Ti with Compute Capability 6.1"

### libraries tensorflow and numba for connecting with GPU

### large download ahead. You'll use the training set, which is a 13GB file MS-COCO dataset https://cocodataset.org/#home.

Researches that involve both vision and languages have attracted great attentions recently as we can witness from the bursting works on image descriptions like image caption and paragraph.Image descriptions aim to generate sentence(s) to describe facts from images in human-level languages. 

## For the Result, the project is divided into two parts i.e Image Captioning + Sonnet generation from Image Caption using LSTM RNN in Python with Keras 

## FINAL OUTPUT OF OUR PROJECT

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/20.jpeg" width="610px" height="530px">

Final code https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/FinalSonnetGenerationCode/finalsonnetrun.ipynb


## SOLUTION ARCHITECTURE

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/18.jpeg" width="600px" height="400px">



We have downloaded the MS-COCO dataset https://cocodataset.org/#home , preprocessed it and it caches a subset of images using Inception V3, trains an encoder-decoder model, and generates captions on new images using the trained model.

Given an image like the example below, our goal is to generate a caption such as
### "a delicious chocolate cake slice".

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/COCO_train2014_000000548913.jpg" width="590px" height="350px">

Image captioning code https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Run.py  Run this code for image caption generation.

To accomplish this, I have used an attention-based model, which enables us to see what parts of the image the model focuses on as it generates a caption.

## Image Caption generation Explanation of code run

1) Importing modules and libraries

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/9.jpeg" width="590px" height="350px">


2) Download and prepare the MS-COCO dataset

GPU USED IS "GeForce GTX 1080 Ti with Compute Capability 6.1"

The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. The code below downloads and extracts the dataset automatically.

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/10.jpeg" width="590px" height="350px">


3) Preprocess the images using InceptionV3

Next, you will use InceptionV3 (which is pretrained on Imagenet) to classify each image. You will extract features from the last convolutional layer.
First, you will convert the images into InceptionV3's expected format by:

    Resizing the image to 299px by 299px
    Preprocess the images using the preprocess_input method to normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train InceptionV3.
    
<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/11.jpeg" width="590px" height="350px">

4) Initialize InceptionV3 and load the pretrained Imagenet weights

Now i have created a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture. The shape of the output of this layer is 8x8x2048. You use the last convolutional layer because you are using attention in this example. You don't perform this initialization during training because it could become a bottleneck.

    You forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
    After all the images are passed through the network, you pickle the dictionary and save it to disk.
Caching the features extracted from InceptionV3 Performance could be improved with a more sophisticated caching strategy, but that would require more code.

The caching will take about 10 minutes to run in Colab with a GPU.
    
<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/12.jpeg" width="590px" height="350px">

5) Preprocess and tokenize the captions

    First, i have tokenize the captions (for example, by splitting on spaces). This gives us a vocabulary of all of the unique words in the data
    Next, i have limit the vocabulary size to the top 5,000 words (to save memory). You'll replace all other words with the token "UNK" (unknown).
    You then create word-to-index and index-to-word mappings.
    Finally, you pad all sequences to be the same length as the longest one.
    
    <img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/13.jpeg" width="590px" height="350px">
    
6) Split the data into training and testing

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/14.jpeg" width="590px" height="350px">
    
    
7) Create a tf.data dataset for training

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/15.jpeg" width="590px" height="350px">

8) Model & Checkpoint

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/17.jpeg" width="590px" height="350px">

9) Training

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/16.jpeg" width="590px" height="350px">

TRY IN YOUR OWN IMAGES

### RESULTS

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/19.jpeg" width="590px" height="400px">


## SONNET GENERATION USING THE CAPTION GENERATED BY THE IMAGE CAPTIONING

Sonnet Generation With LSTM RNN in Python with Keras

## Code for this implementation https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/FinalSonnetGenerationCode/finalsonnetrun.ipynb

TEXT GENERATION USING LSTM  https://github.com/akshitagupta15june/AI_SonnetGeneration


## STEPS USED FOR IMPLEMENTATION

### 1) Let’s start off by importing the classes and functions we intend to use to train our model.

   Next, we need to load the ASCII text for the dataset into memory and convert all of the characters to lowercase to reduce the vocabulary that the network must learn. Create a set of all of the distinct characters in the book, then creating a map of each character to a unique integer.
To know the total characters and total vocab print it.

<img src="https://github.com/akshitagupta15june/AI_SonnetGeneration/blob/master/Images/1.jpeg" width="590px" height="400px">


You can see that there may be some characters that we could remove to further clean up the dataset that will reduce the vocabulary and may improve the modeling process.
We can see that the dataset has under 2,33,972 characters and that when converted to lowercase that there are only 67 distinct characters in the vocabulary for the network to learn.



### 2) As we split up the datset into these sequences, we convert the characters to integers.We could just as easily split the data up by sentences and pad the shorter sequences and truncate the longer ones.

Each training pattern of the network is comprised of 100 time steps of one character (X) followed by one character output (y). When creating these sequences, we slide this window along the whole dataset one character at a time, allowing each character a chance to be learned from the 100 characters that preceded it (except the first 100 characters of course).

<img src="https://github.com/akshitagupta15june/AI_SonnetGeneration/blob/master/Images/2.jpeg" width="590px" height="200px">

Running the code to this point shows us that when we split up the dataset into training data for the network to learn that we have just under 2,33,872 training pattens. This makes sense as excluding the first 100 characters, we have one training pattern to predict each of the remaining characters.



### 3) Now that we have prepared our training data we need to transform it so that it is suitable for use with Keras.

First we must transform the list of input sequences into the form [samples, time steps, features] expected by an LSTM network.
Next we need to rescale the integers to the range 0-to-1 to make the patterns easier to learn by the LSTM network that uses the sigmoid activation function by default.

Finally, we need to convert the output patterns (single characters converted to integers) into a one hot encoding. This is so that we can configure the network to predict the probability of each of the 67 different characters in the vocabulary (an easier representation) rather than trying to force it to predict precisely the next character. Each y value is converted into a sparse vector with a length of 67, full of zeros except with a 1 in the column for the letter (integer) that the pattern represents.

<img src="https://github.com/akshitagupta15june/AI_SonnetGeneration/blob/master/Images/3.jpeg" width="590px" height="200px">


We can now define our LSTM model. Here we define a single hidden LSTM layer with 256 memory units. The network uses dropout with a probability of 20. The output layer is a Dense layer using the softmax activation function to output a probability prediction for each of the 67 characters between 0 and 1.

The problem is really a single character classification problem with 67 classes and as such is defined as optimizing the log loss (cross entropy), here using the ADAM optimization algorithm for speed.

## Giving the image path and storing caption in result variable


<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/22.jpeg" width="590px" height="350px">



### 4) The network is slow to train (about 300 seconds per epoch). Because of the slowness and because of our optimization requirements, we will use model checkpointing to record all of the network weights to file each time an improvement in loss is observed at the end of the epoch. 

We will use the best set of weights (lowest loss) to instantiate our generative model in the next section.We got results, but not excellent results in the previous section. Now, we can try to improve the quality of the generated text by creating a much larger network. We will keep the number of memory units the same at 256, but add a second layer.

Using Larger LSTM Recurrent Neural Network

<img src="https://github.com/akshitagupta15june/AI_SonnetGeneration/blob/master/Images/4.jpeg" width="590px" height="200px">




### Generating Text with an LSTM Network

Generating text using the trained LSTM network is relatively straightforward.

Firstly, we load the data and define the network in exactly the same way, except the network weights are loaded from a checkpoint file and the network does not need to be trained.Also, when preparing the mapping of unique characters to integers, we must also create a reverse mapping that we can use to convert the integers back to characters so that we can understand the predictions.

The simplest way to use the Keras LSTM model to make predictions is to first start off with a seed sequence as input, generate the next character then update the seed sequence to add the generated character on the end and trim off the first character. This process is repeated for as long as we want to predict new characters (e.g. a sequence of 1,000 characters in length).

## We will give text generated by the Image captioning to LSTM MODEL AND THE FINAL SONNET WILL BE GENERATED

# FINAL RESULTS 

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/20.jpeg" width="610px" height="530px">

# ANOTHER RESULT

<img src="https://github.com/akshitagupta15june/ImageSonnetGeneration/blob/main/Image_Captioning/Images/21.jpeg" width="610px" height="530px">



## As Sonnet is a 14 lines poem so in above results we can see that when user entered a caption the poem which is generated in 14 lines have meaningful words.


### Conclusion

project is divided into two parts i.e Image Captioning + Sonnet generation from Image Caption using LSTM RNN in Python with Keras.
In conclusion, two different approaches were used to generate poems. LSTM networks learned the structures of poems and grammatics of english quite well, however lacked the overall meaning. Nevertheless there were some poems that were somewhat reletable. On the other hand seemed mainly to learn some pairs and triplets of words that were common in the training data. In the sampling phase it managed to form some poems that did not exist in the training data and could interpreted as a original and coherent poetry.

Reference links

https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/

https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2


Contributors/Team Members

Akshita Gupta

Anamika Pal

Siddharth Srivastava

Nitish Mangesh Kanlan
    













