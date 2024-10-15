# SDRIS

Reference this [repo](https://github.com/DerrickWang005/CRIS.pytorch) for environmental configuration and dataset preparation

## Preparation

@@ -13,14 +13,14 @@

This implementation only supports **multi-gpu**, **DistributedDataParallel** training, which is faster and simpler; single-gpu or DataParallel training is not supported. Besides, the evaluation only supports single-gpu mode.

To do training of CRIS with multiple GPUs, run:
To do training of MASD with multiple GPUs, run:

```
# e.g., Evaluation on the val-set of the RefCOCO dataset
python -u train.py --config config/refcoco/sdris_r50.yaml
```

To do evaluation of CRIS with 1 GPU, run:
To do evaluation of MASD with 1 GPU, run:
```
# e.g., Evaluation on the val-set of the RefCOCO dataset
CUDA_VISIBLE_DEVICES=0 python -u test.py \
