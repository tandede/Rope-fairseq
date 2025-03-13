cd ../..
fairseq-preprocess \
    --source-lang en --target-lang de \
    --trainpref examples/translation/wmt17_en_de/train \
    --validpref examples/translation/wmt17_en_de/valid \
    --testpref examples/translation/wmt17_en_de/test \
    --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 \
    --workers 20