# Get a song for your mood

![Music](https://github.com/Jorge-Doncel/Song-for-your-mood/blob/master/input/portda.png)

## Table of Contents

- [API](#API)
- [Objective](#Objective)
- [Libraries](#libraries)
- [Resources](#resources)
- [How does it works?](#How-does-it-works?)
- [Data Processing](#data-processing)


## API

Click this link [API songs](https://apisongsface.herokuapp.com/)

## Objective

Nowadays, we have access to an unlimited number of songs of all possible types. Many times, with so much possibility, we do not know where to choose.

Depending on the day, the moment, or our mood, we decided between thousands of songs.

For the days where it is difficult to decide, you can use this API. Upload a photo and, depending on your mood, the API will recommend you a song for your mood.

## Libraries

- Pandas
- Numpy
- cv2 
- PIL
- json
- keras
- matplotlib
- NLTK
- tensorflow
- seaborn
- sklearn
- face_recognition
- glob
- itertools

## Resources

2 datasets where using for train the model. [Fer](https://www.kaggle.com/ahmedmoorsy/facial-expression) dataset and [Jonathan](https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset) train and validation images.

Click [here](https://www.kaggle.com/imuhammad/audio-features-and-lyrics-of-spotify-songs) if you can to go to Spotify dataset

## How does it work?

![funciona](https://github.com/Jorge-Doncel/Song-for-your-mood/blob/master/input/funciona.png)

## Data Processing

First, 2 NN models were train so they differentiate 7 different types of sentiment: angry, disgust, fear, happy, neutral, sad and surprise. We used the model with best accuracy.

Face recognition library were use to detect the face in a photo, take the coordenates, cut the photo, transform to gray scale and resize the image to 48x48 (the model were train with 48x48 image)

In order to select the best song, Spotify database were used. Just english songs were selected. After analize the sentiment of all lyrics with NLTK library, all songs were categorize in 7 different sentiments.

Finally, web scraping and regex were used to get the song url in youtube. 