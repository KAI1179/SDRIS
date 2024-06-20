# RIS


## Preparation

1. Environment
   - [PyTorch](www.pytorch.org) (e.g. 1.10.0)
   - Other dependencies in `requirements.txt`
2. Datasets
   - The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

## Training

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode.

To do training of CRIS with multiple GPUs, run:

```
# e.g., Evaluation on the val-set of the RefCOCO dataset
python -u train.py --config config/refcoco/cris_r50.yaml
```

To do evaluation of CRIS with 1 GPU, run:
```
# e.g., Evaluation on the val-set of the RefCOCO dataset
CUDA_VISIBLE_DEVICES=0 python -u test.py \
      --config config/refcoco/cris_r50.yaml \
      --opts TEST.test_split val-test \
             TEST.test_lmdb datasets/lmdb/refcocog_g/val.lmdb
```

