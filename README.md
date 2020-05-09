# Airplane classifier by using CNN

This is one of the application of online Fastai [course](https://course.fast.ai/videos/?lesson=2). Instead of using the teddy-bear-grizzly-bear classifier like Mr. Jeremy Howard, I try to use a harder application, which is an airplane classifier.

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

I don't have that many airplane pictures to use it as a dataset, I'm not Sam Chui lmao, in fact I have zero airplanes picture in my computer. Thus, I have to collect them from [Google Images](https://images.google.com)

However, unlike using fastai built-in function to scrape the images from Google Images, I use a package ``google_images_download``




