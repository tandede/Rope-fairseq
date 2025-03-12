### Features:

* multi-GPU training on one machine or across multiple machines (data and model parallel)
* fast generation on both CPU and GPU with multiple search algorithms implemented:
  + beam search
  + Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  + sampling (unconstrained, top-k and top-p/nucleus)
  + [lexically constrained decoding](examples/constrained_decoding/README.md) (Post & Vilar, 2018)
* [gradient accumulation](https://fairseq.readthedocs.io/en/latest/getting_started.html#large-mini-batch-training-with-delayed-updates) enables training with large mini-batches even on a single GPU
* [mixed precision training](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-with-half-precision-floating-point-fp16) (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* [extensible](https://fairseq.readthedocs.io/en/latest/overview.html): easily register new models, criterions, tasks, optimizers and learning rate schedulers
* [flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration
* [full parameter and optimizer state sharding](examples/fully_sharded_data_parallel/README.md)
* [offloading parameters to CPU](examples/fully_sharded_data_parallel/README.md)

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
with a convenient `torch.hub` interface:

``` python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

See the PyTorch Hub tutorials for [translation](https://pytorch.org/hub/pytorch_fairseq_translation/)
and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.x)
# pip install fairseq
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained models and examples

We provide pre-trained models and pre-processed, binarized test sets for several tasks listed below,
as well as example training and evaluation commands.

* [Translation](examples/translation/README.md): convolutional and transformer models are available
* [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available

We also have more detailed READMEs to reproduce results from specific papers:

* [XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale (Babu et al., 2021)](examples/wav2vec/xlsr/README.md)
* [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
* [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
* [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
* [Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)](examples/quant_noise/README.md)
* [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
* [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
* [Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)](examples/layerdrop/README.md)
* [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md)
* [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
* [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
* [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
* [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
* [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
* [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
* [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
* [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/README.conv.md)

# Join the fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
