# SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts
This is an official code release of [SP3D](http://arxiv.org/abs/2503.06467) (SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts). 

## Semantic Prompt Generation
### 1. Install FastSAM
```
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
conda create -n FastSAM python=3.9
conda activate FastSAM
cd FastSAM
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```
#### FastSAM Inference
```
python getCameraSam.py pointSAM.yaml
```

### 2. Install SemanticSAM
```
git clone git@github.com:fudan-zvg/Semantic-Segment-Anything.git
cd Semantic-Segment-Anything
conda env create -f environment.yaml
conda activate ssa
python -m spacy download en_core_web_sm
cd ..
git clone git@github.com:facebookresearch/segment-anything.git
cd segment-anything; pip install -e .; cd ../Semantic-Segment-Anything
```

### 3. Download the pretrained model
```
clip_model:<your path>/clip
oneformer_model:<your path>/oneformer
coco_model:<your path>/coco_swin
blip_model:<your path>/blip-image-captioning-large
clipseg_model:<your path>/clipseg-rd16
```


