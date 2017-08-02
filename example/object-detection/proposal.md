# Proposals for object-detection modules

### Algorithms

By stage
- Single stage: SSD,YOLO, RetinaNet
- Two stage: Faster-RCNN, RFCN, MaskRCNN

By data shape
- Immutable data_shape: SSD, YOLO
- Mutable data_shape: Faster-RCNN, RFCN, RetinaNet, MaskRCNN

### Dataset
- Pascal VOC
- COCO
- Imagenet
- Custom dataset Wrapper

### Network Builder
- Anchor generator(CustomOp/function): SSD,YOLO, Faster-RCNN, RFCN, RetinaNet
- Feature extractor(SymbolBlock)
- Box predictor(Conv2D, Dense, Complex): predict box class/location
- Target generator(CustomOp/function calling nd internally): SSD,YOLO, Faster-RCNN, RFCN, RetinaNet
- Loss(gluon.loss): L1, L2, SmoothL1, CE loss, Focal loss
- Anchor converter(CustomOp, function): anchor + prediction = output
- Non-maximum-suppression(src/contrib)

### Matcher(performance critical)
*src/contrib/box_op-inl.h?*

- IOU: overlap between anchors and labels. (How to handle padding?)
- Areas: box area
- Intersection: box intersection
- Clip: clip box to region
- Sampler: generating positive/negative/other samples
- OHEM sampler: Hard negative mining
- More

### Iterator
- Normal iterator(python iterator): SSD/YOLO iterator
- Mutable iterator(wrapper for c++ iter?): batch_size=1, take arbitrary data shape. Usually don't need augmentation.
- Mini-batch iterator: for rcnn variations.

### Trainer
Apply implementations details in each paper

- Initializer (specific init patterns)
- Reshape input after N epochs (YOLO2)
- Warm up / refactor learning rate

### Tester
Allow arbitrary sized input?

### Suger
- Model zoo
- Dataset downloader/loader
- Configuration loader/saver(yaml)
