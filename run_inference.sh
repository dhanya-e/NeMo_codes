#!/usr/bin/env bash

stage=0       # start from 0 if you need to prepare dataset
stop_stage=100

data_root=data/IITM_hindi #IITM <path to audio, text, manifest>

input_manifest=$data_root/test.json #dev.json
asr_model_path=models/ai4b_indicConformer_hi.nemo  # #models/stt_hi_conformer_ctc_large.nemo

out_folder=$data_root/ai4bharat_direct_inference
logfile=$out_folder/asr_output_greedy.log
output_manifest=$out_folder/asr_output_manifest.json


#dataset preparation
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    python local/get_hindi_data.py \
        --data_root $data_root \
        --data_sets test  
fi

# #transcribing input audio in a batch
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
     mkdir -p $out_folder
     
     python scripts/transcribe_speech.py \
        model_path=$asr_model_path \
        dataset_manifest=$input_manifest \
        batch_size=1 \
        amp=True \
        calculate_rtfx=True \
        use_cer=True \
        decoder_type=ctc \
        output_filename=$output_manifest | tee $logfile
fi

# inference with LM
logfile=$out_folder/inference_withLM.log
output_manifest=$out_folder/asr_output_manifest_withLM.json
#lm_path=ata/IITM_hindi/kenLM_model/base_lm.arpa  #data/IndicVoices/LM_training/kenLM_model/base_lm.arpa

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p $out_folder
    
    python scripts/eval_beamsearch_ngram_ctc.py \
      nemo_model_file=$finetuned_model_path \
      input_manifest=$input_manifest \
      kenlm_model_file=$lm_path \
      beam_width=[128] \
      beam_alpha=[1.0] \
      beam_beta=[1.0,0.5] \
      preds_output_folder=$data_root/dhan_ctc_out/ \
      decoding_strategy="pyctcdecode" \
      decoding_mode="beamsearch_ngram" | tee $logfile
fi