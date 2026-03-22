#!/usr/bin/env bash

# https://github.com/huggingface/tokenizers/issues/690
#https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/configs.html#fine-tuning-configurations

data_root=/home/dhanya/NeMo/egs/hindi_asr/data/IITM_hindi #IITM

# tokenizer 
vocab_generate="" # flag for new vocab generation

if [ -z ${vocab_generate} ]; then
    echo "Fine-tuning Nemo models using old vocab on $data_root"
    # fine-tuning with pre-traning vocab on new target dataset
    python scripts/speech_to_text_finetune.py \
        model.train_ds.manifest_filepath=$data_root/train.json \
        model.validation_ds.manifest_filepath=$data_root/test.json \
        trainer.devices=-1 \
        trainer.accelerator=gpu \
        trainer.max_epochs=30 \
        +init_from_nemo_model=models/stt_hi_conformer_ctc_large.nemo

else
    echo "new vocab"
    # bpe based vocab generation for the fine-tune dataset
    python scripts/process_asr_text_tokenizer.py \
      --manifest=$data_root/dev.json \
      --data_root=$data_root/tokenizers/ \
      --vocab_size=256 \
      --tokenizer=spe \
      --spe_type=bpe \
      --spe_character_coverage=1.0 \
      --log
    
    # fine-tuning
    python scripts/speech_to_text_finetune.py \
        model.train_ds.manifest_filepath=$data_root/dev.json \
        model.validation_ds.manifest_filepath=$data_root/test.json \
        model.tokenizer.update_tokenizer=True \
        model.tokenizer.dir=$data_root/tokenizers/tokenizer_spe_bpe_v256/ \
        model.tokenizer.type=bpe \
        trainer.devices=-1 \
        trainer.accelerator=gpu \
        trainer.max_epochs=30 \
        +init_from_nemo_model=models/stt_hi_conformer_ctc_large.nemo
fi