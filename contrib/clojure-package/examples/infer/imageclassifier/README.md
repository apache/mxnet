# imageclassifier

Run image classification using clojure infer package.

## Installation

`lein install`

## Usage

```
$ chmod +x scripts/get_resnet_18_data.sh
$ ./scripts/get_resnet_18_data.sh
$ lein uberjar
$ java -jar target/imageclassifier-0.1.0-SNAPSHOT-standalone.jar --help
$ java -jar target/imageclassifier-0.1.0-SNAPSHOT-standalone.jar \
    -m models/resnet-18/resnet-18 -i images/kitten.jpg -d images/
```
