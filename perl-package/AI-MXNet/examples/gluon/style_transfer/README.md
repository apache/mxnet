This directory provides AI::MXNet Implementation of MSG-Net real time style transfer, https://arxiv.org/abs/1703.06953

### Stylize Images Using Pre-trained MSG-Net
Download the pre-trained model
        ```
        ./get_data.sh
        ```

Test the model
        ```
        ./style_transfer.pl --content-image <path or url> --style-image < path or url> --content-size 512
        ```

More options:

        * --content-image: path or url to content image you want to stylize.
        * --style-image:   path or url to style image.
        * --model:         path to the pre-trained model to be used for stylizing the image if you use your custom model
        * --output-image:  path for saving the output image, default is 'out.jpg'
        * --content-size:  the content image size to test on, default is 512 pixels for the shorter side,
                             decrease the size if your computer is low on RAM and the script fails

<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/1.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/2.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/3.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/4.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/5.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/6.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/7.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/8.jpg" width="260px" />
<img src ="https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/example/style_transfer/images/9.jpg" width="260px" />
