# 1. Pretrain

### 1.1. Download data

```bash
python bin/download.py --out=data/pretrain tokens_v1_train_val.tar.gz
tar -xzvf data/pretrain/tokens_v1_train_val.tar.gz -C data/pretrain/
mv data/pretrain/mp_oqmd_nomad_cifs_semisymm_Z_props_2/* data/pretrain/
rm -r data/pretrain/mp_oqmd_nomad_cifs_semisymm_Z_props_2
python bin/download.py --out=data/pretrain starts_v1_train.pkl
python bin/download.py --out=data/pretrain starts_v1_val.pkl
```

### 1.2. Pretrain
Here we pretrain a small model (two blocks) for only two epochs.
```bash
python bin/train.py --config=cvec_playground/pretrain.yaml
```

# 2. Finetuning with conditioning vectors

### 2.1 Prepare data
First, we copy the dataset for pretraining as the text part for the finetuning dataset. 
In practice, however, the finetuning dataset may be (much) smaller than the pretraining one.

```bash
cp -r data/pretrain data/finetuning
```

Now we need to associate each text with a conditioning vector.
As a layman for chemistry, I assume each vector contains two real numbers sample from N(0,1).
This can be done by
```bash
python cvec_playground/gen_fake_cvec.py --dir=data/finetuning --n-cvec=2
```
Here `--n-cvec` specifies the length of the conditioning vectors.
You will find `cvec_train.npz` and `cvec_val.npz` created under `data/finetuning`. 

Replacing `gen_fake_cvec.py` with a chemically meaningful process would be the hardest step for applications.

### 2.2 Finetuning
First, we must copy the pretrained model as the starting point of finetuning:
```bash
mkdir out/finetuning
cp out/pretrain/ckpt.pt out/finetuning/
```

Eventually, we are able to finetune the pretrained model with the generated conditioning vectors:
```bash
python bin/train.py --config=cvec_playground/finetuning.yaml
```

# 3. Inference
At inference time, we can use the pretrained model without conditioning vectors:
```bash
python cvec_playground/inference.py --model=out/pretrain --id=silicon_dioxide --text=SiO2 
```

Or we can use the finetuned model with conditioning vectors:
```bash
python cvec_playground/inference.py --model=out/finetuning --id=silicon_dioxide --text=SiO2 --cvec 0.1 0.2 
```

