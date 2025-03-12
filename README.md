# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/tandede/Rope-fairseq.git
cd Rope-fairseq
pip install --editable ./

```
# 修改部分
## 主要修改修改了两个文件
### 1. 修改了Rope-fairseq/fairseq/modules/multihead_attention.py
#### 添加了rope函数接口，使其能够融入transformer预训练中
### 2. 修改了Rope-fairseq/fairseq/modules/rotary_positional_embedding.py
#### 添加了几个不同的编码形式，分别进行预训练对比
### 3.将Rope-fairseq/fairseq/models/transformer中下的几个python文件，将原始绝对位置编码进行注释，不进行添加
# Pre-trained models and examples
``` bash
# 准备工作
pip install tensorboardX
pip install sacremoses
```
## 1.下载安装
``` bash
git clone https://github.com/tandede/Rope-fairseq.git
cd Rope-fairseq
pip install --editable ./
```
## 2. 数据下载、预处理
``` bash
cd /yourpath/Rope-fairseq/examples/translation
 ./prepare-wmt14en2de.sh
```
## 3.Binarize the dataset
``` bash
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref /yourpath/Rope-fairseq/examples/translation/wmt17_en_de/train \
    --validpref /yourpath/Rope-fairseq/examples/translation/wmt17_en_de/valid \
    --testpref /yourpath/Rope-fairseq/examples/translation/wmt17_en_de/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20
 ```
  ## 4. Train
  ### 单卡训练
  ``` bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    /yourpath/data-bin/wmt17_en_de  \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
  ```
  ### 多卡训练
  ``` bash
  CUDA_VISIBLE_DEVICES=0,1  fairseq-train \
    /yourpath//data-bin/wmt17_en_de \
    --distributed-world-size 2
    --distributed-num-procs 2
    --arch transformer_wmt_en_de --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000  \
    --lr 5e-4 --min-lr 1e-09 --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 --dropout 0.3 --weight-decay 0.0001 --max-tokens 4096   \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --update-freq 2 |tee exp3.log
  ```
  ## 5. Test
  ### 平均检查点
  ``` bash
python scripts/average_checkpoints.py \
    --inputs /yourpath/checkpoints \
    --num-epoch-checkpoints  5 --output averaged_model.pt
  ```
  ### 生成测试文件
  ``` bash
  CUDA_VISIBLE_DEVICES=0 python generate.py \
    data-bin/wmt17_en_de --path /yourpath/averaged_model.pt \
    --remove-bpe --beam 4 --batch-size 64 --lenpen 0.6 \
    --max-len-a 1 --max-len-b 50|tee generate.out
  ```
  ### 加点
  ```bash
  grep ^T generate.out | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.ref

  grep ^H generate.out |cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > generate.sys
  ```  
  ### 计算BLEU
  ``` bash
  python /yourpath/fairseq_cli/score.py \
    --sys generate.sys \
    --ref generate.ref
  ```


``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
