# Image Super-Resolution by Learning Weighted Convolutional Sparse Coding
This repository is for LW-CSC.

## Dependencies
* Python 2 (Recommend to use [Anaconda](https://www.anaconda.com/distribution/#linux))
* [Pytorch 1.0.1](https://pytorch.org/)
* NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
* Python packages: pip install xxx
## Train
### Prepare training data
1. Download the 291 images ([Baidu Netdisk](https://pan.baidu.com/s/1bEajYJm_X5aVoXdS3RcbQg) psw:ryjr), and place them in './data' folder.
2. cd to './data', and run `generate_train.m` to generate training data.

### Begin to train
1. (optional) Download the model for our paper and place it in './pretrained'.
2. Run the following script to train.

```bash
bash train.sh
```

## Test
1. Run the following script to evaluate.

   ```python
   python evaluate.py
   ```

## Citation

If you use any part of this code in your research, please cite our paper:

```latex
@article{lwcsc2021,
  title={Image Super-Resolution by Learning Weighted Convolutional Sparse Coding},
  author={He, Jingwei and Yu, Lei and Liu, Zhou and Yang, Wen},
  journal={Signal, Image and Video Processing},
  volume={x},
  number={x},
  pages={xx--xx},
  year={2021},
  publisher={Springer}
}
```
