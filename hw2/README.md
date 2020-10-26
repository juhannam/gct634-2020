# Homework #2: Music Genre Classification [Colab Notebook](https://colab.research.google.com/drive/1-9SrI7M440hQ6F2Q7KLiBrFjzSTfyG57?usp=sharing)
Music genre classification is an important task that can be used in many musical applications such as music search or recommender systems. Your mission is to build your own Convolutional Neural Network (CNN) model to classify audio files into different music genres. Specifically, the goals of this homework are as follows:

* Experiencing the whole pipeline of deep learning based system: data preparation, feature extraction, model training and evaluation
* Getting familiar with the CNN architectures for music classification tasks
* Using Pytorch in practice

## Dataset
We use the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset which has been the most widely used in the music genre classification task. 
The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. 
For this homework, we are going to use a subset of GTZAN with only 8 genres. You can download the subset from [this link](https://drive.google.com/file/d/1J1DM0QzuRgjzqVWosvPZ1k7MnBRG-IxS/view?usp=sharing).

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  
```
$ tar zxvf gtzan.tar.gz
$ cd gtzan
$ ls *
split:
test.txt  train.txt

wav:
blues     classical country   disco     hiphop    jazz      metal     pop       reggae    rock
$ cd ..      # go back to your home folder for next steps
```


## Training CNNs from Scratch
The baseline code is provided so that you can easily start the homework and also compare with your own algorithm.
The baseline model extracts mel-spectrogram and has a simple set of CNN model 
that includes convolutional layer, batch normalization, maxpooling and dense layer.

### Question 1: Implement a CNN based on a given model specification
An architecture of CNN will be provided. Implement a CNN following the architecture.

## Exploiting Prior Knowledge using Pre-trained Models
Someone who knows how to play acoustic guitars might be better at playing electric guitars than who never played a guitar.
Here, we will use pre-trained models from [`musicnn`](https://github.com/jordipons/musicnn) (pronounced as "musician"), which includes CNNs already trained on a large amount of songs.

### Question 2: Train a 2-layer MLP using the extracted features from the Pre-trained Model
Create 2-layer MLP model and train the model using the extracted features.


## Improving Algorithms [[Leader Board]](https://docs.google.com/spreadsheets/d/1bzkMFeXABTae7kDJG6QCU_qnP1ppJDoNQLgGz3ksJu0/edit?usp=sharing)
### Question 3: Improve performances of models
Now it is your turn. You should improve the baseline code with your own algorithm. There are many ways to improve it. The followings are possible ideas: 

* The first thing to do is to segment audio clips and generate more data. The baseline code utilizes the whole mel-spectrogram as an input to the network (e.g. 128x1287 dimensions). Try to make the network input between 3-5 seconds segment and average the predictions of the segmentations for an audio clip.

* You can try training a model using both mel-spectrograms and features extracted using the pre-trained models. The baseline code is using a pre-trained model trained on 19k songs, but `musicnn` also has models trained on 200k songs! Try using the model giving `model='MSD_musicnn'` option on feature extraction.

* You can try 1D CNN or 2D CNN models and choose different model parameters:
    * Filter size
    * Pooling size
    * Stride size 
    * Number of filters
    * Model depth
    * Regularization: L2/L1 and Dropout

* You should try different hyperparameters to train the model and optimizers:
    * Learning rate
    * Model depth
    * Optimizers: SGD (with Nesterov momentum), Adam, RMSProp, ...

* You can try different parameters (e.g. hop and window size) to extract mel-spectrogram or different features as input to the network (e.g. MFCC, chroma features ...). 

* You can also use ResNet or other CNNs with skip connections. 

* Furthermore, you can augment data using digital audio effects.


## Deliverables
You should submit your Python code (`.ipynb` or `.py` files) and homework report (.pdf file) to KLMS. The report should include:
* Algorithm Description
* Experiments and Results
* Discussion

## Note
The code is written using PyTorch but you can use TensorFlow if you want for question 3.

## Credit
Thie homework was implemented by Jongpil Lee, Soonbeom Choi and Taejun Kim in the KAIST Music and Audio Computing Lab.
