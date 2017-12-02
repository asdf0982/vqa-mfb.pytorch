# Multi-modal Factorized Bilinear Pooling (MFB) for VQA
This is an unofficial and Pytorch implementation for [Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering](http://openaccess.thecvf.com/content_iccv_2017/html/Yu_Multi-Modal_Factorized_Bilinear_ICCV_2017_paper.html) and [Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering](https://arxiv.org/abs/1708.03619).

![Figure 1: The MFB+CoAtt Network architecture for VQA.](https://github.com/asdf0982/vqa-mfb.pytorch/raw/master/imgs/MFB-github.png)

The result of MFB-baseline and MFH-baseline can be replicated.(Not able to replicate MFH-coatt-glove result, maybe a devil hidden in detail.)

The author helped me a lot when I tried to replicate the result. Great thanks.

The official implementation is based on pycaffe is available [here](https://github.com/yuzcccc/vqa-mfb).
## Requirements
Python 2.7, pytorch 0.2, torchvision 0.1.9, [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)
## Result
|   Datasets\Models    | MFB | MFH  | MFH+CoAtt+GloVe (FRCN img features) |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| VQA-1.0   |58.75%    | 59.15%  | **68.78%** |

- MFB and MFH refer to MFB-baseline and MFH-baseline, respectively.
- The results of MFB and MFH are trained with train sets, tested with val sets, using ResNet152 pool5 features. The result of MFH+CoAtt+GloVe is trained with train+val sets, tested with test-dev sets.

![Figure 2: MFB-baseline result](https://github.com/asdf0982/vqa-mfb.pytorch/raw/master/imgs/mfb_baseline.png)

![Figure 3: MFH-baseline result](https://github.com/asdf0982/vqa-mfb.pytorch/raw/master/imgs/mfh_baseline.png)
## Training from Scratch
`$ python train_*.py`
- Most of the hyper-parameters and configrations with comments are defined in the `config.py` file.
- Pretrained GloVe word embedding model (the spacy library) is required to train the mfb/h-coatt-glove model. The installation instructions of spacy and GloVe model can be found [here](https://spacy.io/models/en#section-en_vectors_web_lg).

## Citation
If you find this implementation helpful, please consider citing:
```
@article{yu2017mfb,
  title={Multi-modal Factorized Bilinear Pooling with Co-Attention Learning for Visual Question Answering},
  author={Yu, Zhou and Yu, Jun and Fan, Jianping and Tao, Dacheng},
  journal={IEEE International Conference on Computer Vision (ICCV)},
  year={2017}
}

@article{yu2017beyond,
  title={Beyond Bilinear: Generalized Multi-modal Factorized High-order Pooling for Visual Question Answering},
  author={Yu, Zhou and Yu, Jun and Xiang, Chenchao and Fan, Jianping and Tao, Dacheng},
  journal={arXiv preprint arXiv:1708.03619},
  year={2017}
}
```