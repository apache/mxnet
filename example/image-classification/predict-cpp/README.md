# Image Classification Example of C++
This is a simple predictor which shows how to use c api for image classfication.

It uses opencv for image reading

# How to Use

## Build
1. Edit image-classification-predict.cc file, change the following lines to your model paths:
  ```bash
    // Models path for your model, you have to modify it
    BufferFile json_data("model/Inception/Inception_BN-symbol.json");
    BufferFile param_data("model/Inception/Inception_BN-0039.params");
  ```

2. Edit synset file path if you have it:
  ```bash
    // Synset path for your model, you have to modify it
    std::vector<std::string> synset = LoadSynset("model/Inception/synset.txt");
  ```
  
3. You may also want to change the image size and channels:  
  ```bash
    // Image size and channels
    int width = 224;
    int height = 224;
    int channels = 3;
  ```
  
4. Simply just use our Makefile to build:
  ```bash
  make
  ```

## Usage
* Run:
  ```bash
  ./image-classification-predict apple.jpg
  ```
The only parameter is the path of the test image.  

## Tips
* The model used in the sample can be downloaded here:
http://pan.baidu.com/s/1sjXKrqX

* If you don't run it in the mxnet root path, maybe you will need to copy lib folder here.

# Author
* **Xiao Liu**

* E-mail: liuxiao@foxmail.com

* Homepage: [www.liuxiao.org](http://www.liuxiao.org/)

# Thanks
* pertusa (for Makefile and image reading check)

* caprice-j (for reading function)

* sofiawu (for sample model)

* piiswrong and tqchen (for useful coding suggestions)


