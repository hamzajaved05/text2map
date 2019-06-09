# texttomap (Under testing)
Text-based landmark localization and place recognition project for Perception and Learning for Robotics course

Note: The files are not yet arranged and formatted and hence there will some issues with filenames and their paths.
Note: The final Triplet network to combine the embeddings of text based features (ParkhiNet) and the Net Vlad embeddings is yet to be added along with its results. The network is however available in code (see file details) for training. The only thing remaining is the mining of hard positives and hard negatives ofr the process. Right now it mines randomly and converges very slowly.
Note: Link for the generated data to be added


The goal of this project was to leverage local text based features to improve the accuracy of global descriptors using deep learning. The approach taken is explained as following:


# 1. Non-Learning Based


Initially, a non-learning based approach was used for place recognition using the text based network. The intuition behind this was to understand the challenges of non-learning based methods in text matching and to come up with strategies to tackle these issues. The outputs (text from images) were generated from a text recognition agent Deeptext Spotter (Busta et al. 17'). The outputs were processed to only include alphanumeric characters. 

## 1.1 Metric


The distance metric for comparison between two strings of text was chosen as Levenshteins' distance. The distance defines the number of manipulations needed to convert one string to the other. The metric makes intuitive sense however there are still some issues with the metric that can not be handled as simply as that. These are discussed in the limitations of non-learning based approach section. The metric is defined in code in util/matching.py

## 1.2 Matching

The metric outputs a score based on the individual matching of two strings and this has to be adapted to match a query image with *n* number of text instances with an image in the databse containing *m* number of text instances. The variation in the number of instances makes the comparision challenging. In order to perform ths, we need a comparison tool which does not fail down due to number of text instances. An example of this would be when we simple add the score for each individual matched text. The comparison tool breaks down as an perfectly matched image having a good score but small value of  *m* will be clearly outscored by another false positive which has accumulated a big score due to high *m*. This is performed in the main executable named hc_main.py

## 1.3 Scoring words

One of the challenges in the implementation of scene recognition using text is how to tackle frequent word instances that are found everywhere such as *taxi*, *shop* etc. The technique implemented was to somehow collect all of these words and to give less weightage to them so that these instances do not throw off the correct guesses. In practice, this does not really affect the performance as the scored words change the score consistently among all of the potential candidates for place recognition. It just rescales the confidence of the image by some value.

## 1.4 Inaccuracy of text recognition agent

The output of even the state of the art text recognition agent is fairly inaccurate due to the randomness and the different patterns of text in natural scenes. Some of the most common misdetections was of windows being labelled as *EEEEE* or *DDDDD*.

### 1.4.1 Usable wrong instances

Despite being misdetected or misrecognized, we were still able to use these instances to perform place recognition. The reason for that is if a particularly styled window is always misrecognized as *EDE* in multiple images, we can use the consistency of the text recognition agent to use that in matching.

### 1.4.2 Unusable instances

As explained in the prior section, some of the data, although misrecognized was still usable but it all depended upon the data being consistent throughout the images. The instances should also be distinct so that the dataset was not crowded by misrecognitions which would ultimately affect the matching.


## Challenges of Non-Learning Based Method

### Metric 
As defined in the above section, the metric chosen was Levenshteins metric. The metric measure the amount of manipulations needed to convert one string to the other (symmetric metric). The issue with this is that these manipulations are too simple and without context for our case. For example, let our query text be *cat* and we have a set of strings *cot* and *cam*. From contextual viewpoint we can infer that the *cot* string should score higher than *cam* which is clearly a different word. But the metric scores both at the same distance from *car*. Another aspect to compare is the probability of misidentification of certain english letters (such as *o* vs *0* and *o* vs *a*). This knowledge can be incorporated in the system as a prevalent bias of our inputs and can be leveraged to improve the accuracy of models.

# Learning Based Approach for Visual text 
To be updated here.
Training file training.py

# Combining NetVlad with Visual text
To be updated here
Training file train.py
