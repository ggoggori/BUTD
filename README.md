# BUTD
Implementation of "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering"

* https://github.com/poojahira/image-captioning-bottom-up-top-down 코드를 참고
* AI HUB에서 제공하는 한국어 MS COCO Image Captioning 학습이 가능
* SentencePiece tokenizer 사용



## Structure
```bash
.
|-- BEST_5checkpoint_5_cap_per_img.pth.tar
|-- bottom-up_features
|   |-- make_cocolist.py
|   |-- train_ids.pkl
|   |-- trainval_36
       |--trainval_resnet101_faster_rcnn_genome_36.tsv
|   |-- tsv.py
|   |-- utils.py
|   `-- val_ids.pkl
|-- caption.py
|-- checkpoint_5_cap_per_img.pth.tar
|-- create_input_files.py
|-- data
|   |-- coco_kor.json
|   |-- train2014
|   `-- val2014
|-- datasets.py
|-- eval.py
|-- feature_extract
|   |-- modeling_frcnn.py
|   |-- preprocessing_image.py
|   `-- utils.py
|-- ko.bin
|-- models.py
|-- requirements.txt
|-- train.py
`-- utils.py
```

## Data preparation
1. "./data"와 "./final_dataset" 폴더를 만든다.

2. MS COCO 이미지를 다운로드 하고 ./data folder에 옮긴다.(AI Hub에서 제공하는 ms coco caption json파일도 함께)

3. https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip 에서 image feature를 다운로드하고, bottom-up_features 폴더에 알집을 푼다.

4.
```bash
python bottom-up_features/tsv.py
```

5.
```bash
python create_input_files.py
```
## train
* 모델을 훈련시킵니다.
* 훈련에 필요한 데이터셋 경로, tokenizer 등과 같은 파라미터는 train.py 내에서 설정 가능합니다.
```bash
python train.py
```
* Embedding layer에 사전학습된 Word Embedding(fasttext)를 사용하려면, https://github.com/Kyubyong/wordvectors 에서 Pretrained Embedding 값을 내려받고, 최상위폴더에 ko.bin을 위치시켜야 합니다.
## eval
* 모델을 평가합니다.
* 평가에 필요한 파라미터는 eval.py에서 설정 가능합니다.
```bash
python eval.py
```
## caption
* 자신이 가지고 있는 이미지로, caption을 생성할 수 있습니다.
```bash
python caption.py --model ./checkpoint_5_cap_per_img.pth.tar --img ./sample.jpg
```
