CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py \
    data-bin/wmt17_en_de --path averaged_model.pt \
    --remove-bpe --beam 4 --batch-size 64 --lenpen 0.6 \
    --max-len-a 1 --max-len-b 50|tee generate.out