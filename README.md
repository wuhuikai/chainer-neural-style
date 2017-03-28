# chainer neural-style & fast-neural-style
Chainer implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
and [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/). 

## neural style
### Download VGG-19
Download the original [VGG-19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md) and then convert it to Chainer model:
```bash
python caffe_model_2_pickle.py
```
**Or** Download the converted VGG-19 Chainer model [here](https://drive.google.com/open?id=0Bybnpq8dvwudM0U3enFsYV9waWM).
### Now you're an artist!!
```
 python neural_style.py --content_image [Content Image] --style_images [Style Image;Style Image;...] 
```
*Note*: `python neural_style -h` for more details.
### Your Gallery
* Content Image
![](images/towernight.jpg)
* Style Image
![](images/Starry_Night.jpg)
* Result
  * `--original_color False --style_color False`
  ![](images/result/towernight_with_style(s)_Starry_Night.png)
  * `--original_color True`
  ![](images/result/OrigColor_towernight_with_style(s)_Starry_Night.png)
  * `--style_color True`
  ![](images/result/StyleColor_towernight_with_style(s)_Starry_Night.png)

* Content Image
![](images/towernight.jpg)
* Style Images
![](images/Starry_Night.jpg)
![](images/the_scream.jpg)
* Result Image
![](images/result/hrbrid_towernight_with_style(s)_Starry_Night_the_scream.png)
