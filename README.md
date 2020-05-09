# Airplane classifier by using CNN

This is one of the application of online Fastai [course](https://course.fast.ai/videos/?lesson=2). Instead of using the teddy-bear-grizzly-bear classifier like Mr. Jeremy Howard, I try to use a harder application, which is an airplane classifier. I use the CNN with resnet34 structure to train the model. As it turns out the accuracy of testing data is 95%, which is pretty great.

My intension being, try to classify some easy distinct planes, like B747 and B737, which is very easy to identify even you are a noob. If the model goes well, we can go ahead and do harder one, like B777-300ER and A350, see how it goes. 


However, I am not gonna do B747-B737 classifier, this is god damn easy lmao, it's like stealing a toy from a baby, with no parents watching, which is seriously fk up and it's is just a metaphor.

My plane classifier is going to identify A340, A380, and B747, and this is what 3 planes looks like.


### Airbus A340

![Airbus A340](https://d1a2ot8agkqe8w.cloudfront.net/web/2019/07/a340-300-air-france_78200.jpg)


### Airbus A380

![alt text](https://upload.wikimedia.org/wikipedia/commons/0/09/A6-EDY_A380_Emirates_31_jan_2013_jfk_%288442269364%29_%28cropped%29.jpg)


### Boeing 747

![alt text](https://img.ruten.com.tw/s2/3/b2/46/21815878559302_909.jpg)

As we can see, those planes have some differences and similarity. Differences such as A340 has no second deck, B747 has half second deck, A380 has full second deck. Similarity such as the engines of three types of planes are on their wings. And there are something I called "mixed signal", there are some features that two plans has in common, and the other is different, such as A340 and B747 has similar wingtip, and A380 is different, which will definitely increase the difficulty of the identification.

## Data collection

I don't have that many airplane pictures to use it as a dataset, I'm not Sam Chui lmao, in fact I have zero airplane picture in my computer. Thus, I have to collect them from [Google Images](https://images.google.com)

However, unlike using fastai built-in function to scrape the images from Google Images, I use a package ``google_images_download``, which is extremely easy to use and much stable than the built-in function of fastai.

I gather 150 pictures of A340, 400 pictures of A380, and 400 pictures of B747. The reason why I gather 150 instead of 400 pictures of A340 is that, Airbus A340 is not a very well-saled plane for Airbus. Airbus only make no more than 400 A340 ever since 1993, thus I can't get that much A340 pictures, if I really did, Google Images will end up give me other planes than A340, or worse, A380 since it is also a 4 engine aircraft.

## Training process

I use resnet34 structure to run the CNN model due to the accuracy will usually get better, and with the power of TPU that Google Colab provide, the whole algorithm will finish within 30 minutes, both first training stage, and final tuning stage.

And as we all know, if you search a picture in Google Image, sometimes you will get incorrect result, and the image will be more incorrect as you scroll down the result page, thus, we may grab some incorrect labeled photo. As a result, I have to find a way to delete those wrong-labeled picture.

Here's is my plan, first, the training process is pre-train a model, and find out the image with highest loss of all the training set and then we can go ahead and delete it

Luckly, fastai develope a very efficient way to do this, we can simply use an interative widget, click a delete button, and simple remove the image from imageDataBunch, code as below.

```python
from fastai.widgets import *

ds, idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Train)

ImageDeleter(ds, idxs)
```

But unfortuanely, I run the code on Google Colab since I love their TPU, it's so good lmao, but I can't use imagedeleter since Colab won't allow using widget except for ipython widget, so I have I manually delete those high loss images.

After cleaning it, we can finally tuning the learning rate, only the learning rate, at this point, we don't have to change every parameter to make my model better, CNN will take care of pretty much everything with a great learning rate and certain amount of epoch.


## Predictability

As it turns out the result is pretty good, testing data has a 95% accuracy, which is exacly what I expected, these are three easy identified planes, not Mark Wahlberg and Matt Damon.

![Confusion Matrix](https://i.imgur.com/bM70UHn.png)

I try to use other pictures, actually three pictures, and it identify them correctly, just like clockwork.


## Conclusion

This is a very interesting and exciting project, with Fastai code I can easily build up a CNN, and get a very good model, probably not the best, but it is good enough.

I think if I really want to identify hard identify planes, like B737-max, A321, A321-neo, I have to really decide which images will be put in the dataset, and try to catch the differences between those planes, such as wingtips or cockpit windows.

Anyway this is a very good application of put CNN in to production, I have a blast to use fastai package and online courses.











