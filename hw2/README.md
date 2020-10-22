# Homework #2: Music Genre Classification
Music genre classification is an important task that can be used in many musical applications such as music search or recommender systems. Your mission is to build your own Convolutional Neural Network (CNN) model to classify audio files into different music genres. Specifically, the goals of this homework are as follows:

* Experiencing the whole pipeline of deep learning based system: data preparation, feature extraction, model training and evaluation
* Getting familiar with the CNN architectures for music classification tasks
* Using Pytorch in practice

## Dataset
We use the [GTZAN](http://marsyas.info/downloads/datasets.html) dataset which has been the most widely used in the music genre classification task. 
The dataset contains 30-second audio files including 10 different genres including reggae, classical, country, jazz, metal, pop, disco, hiphop, rock and blues. 
For this homework, we are going to use a subset of GTZAN with only 8 genres. You can download the subset from [this link](https://drive.google.com/file/d/1rHw-1NR_Taoz6kTfJ4MPR5YTxyoCed1W/view?usp=sharing).

Once you downloaded the dataset, unzip and move the dataset to your home folder. After you have done this, you should have the following content in the dataset folder.  

```
$ cd gtzan
$ ls 
blues disco metal ... rock train_filtered.txt valid_filtered.txt test_filtered.txt
$ cd ..      # go back to your home folder for next steps
```

## Requirements 
Before you run baseline code you will need PyTorch. Please install PyTorch from [this link](https://pytorch.org/get-started/locally/).
We will use PyTorch 1.0 because it is the first official version.

* Python 3.7 (recommended)
* Numpy
* Librosa
* PyTorch 1.0

## Baseline Code
The baseline code is provided so that you can easily start the homework and also compare with your own algorithm.
The baseline model extracts mel-spectrogram and has a simple set of CNN model 
that includes convolutional layer, batch normalization, maxpooling and dense layer.

Baseline code contains following Python files.
* hparams.py: sets hyper parameters for feature extraction and training
* feature_extraction.py: extracts mel-spectrogram using Librosa and stores them in the './feature' folder
* data_manager.py: loads saved features and convert them into PyTorch DataLoader.
* models.py: contains neural network model.
* train_test.py: trains model and tests it for genre classification.

Once you download the files, run the feature extraction first.
```
$ python feature_extraction.py
```

If it runs successfully it will create "feature" folder with "train", "valid" and "test" sub folders.
Then you can run the training code and it will display the result like below
```
$ python train_test.py
...
[Epoch 22/100] [Train Loss: 0.2772] [Train Acc: 0.9263] [Valid Loss: 1.4282] [Valid Acc: 0.5133]
[Epoch 23/100] [Train Loss: 0.3096] [Train Acc: 0.9263] [Valid Loss: 1.4084] [Valid Acc: 0.4867]
Epoch    23: reducing learning rate of group 0 to 1.6000e-05.
[Epoch 24/100] [Train Loss: 0.2849] [Train Acc: 0.9263] [Valid Loss: 1.3976] [Valid Acc: 0.4933]
[Epoch 25/100] [Train Loss: 0.3764] [Train Acc: 0.9008] [Valid Loss: 1.4730] [Valid Acc: 0.4867]
[Epoch 26/100] [Train Loss: 0.3162] [Train Acc: 0.9178] [Valid Loss: 1.3281] [Valid Acc: 0.5533]
[Epoch 27/100] [Train Loss: 0.2639] [Train Acc: 0.9292] [Valid Loss: 1.3755] [Valid Acc: 0.5133]
Epoch    27: reducing learning rate of group 0 to 3.2000e-06.
Training Finished
Test Accuracy: 63.44%
```

## Improving Algorithms
Now it is your turn. You should improve the baseline code with your own algorithm. There are many ways to improve it. The followings are possible ideas: 

* The first thing to do is to segment audio clips and generate more data. The baseline code utilizes the whole mel-spectrogram as an input to the network (e.g. 128x1287 dimensions). Try to make the network input between 3-5 seconds segment and average the predictions of the segmentations for an audio clip.
* You can try 1D CNN or 2D CNN models and choose different model parameters:
    * Filter size
    * Pooling size
    * Stride size 
    * Number of filters
    * Model depth
    * Regularization: L2/L1 and Dropout

* You should try different hyperparameters to train the model and optimizers:
    * Learning rate
    * Patience value
    * Decreasing factor of learning rate 
    * Minibatch size
    * Model depth
    * Optimizers: SGD (with Nesterov momentum), Adam, RMSProp, ...

* You can try different parameters (e.g. hop and window size) to extract mel-spectrogram or different features as input to the network (e.g. MFCC, chroma features ...). 

* You can also use ResNet or other CNNs with skip connections. 

* Furthermore, you can augment data using digital audio effects.


## Deliverables
You should submit your Python code (.py files) and homework report (.pdf file) to KLMS. The report should include:
* Algorithm Description
* Experiments and Results
* Discussion

## Notes
* You can you merge training and validation sets into a single training set (as in HW1). However, you should report both validation and test accuracy to prove that you chose the best model without using the test set.  


## Credit
Thie homework was implemented by Jongpil Lee and Soonbeom Choi in the KAIST Music and Audio Computing Lab.
