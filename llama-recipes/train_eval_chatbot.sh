export CUDA_VISIBLE_DEVICES=0,1,2,3


DIR=your_output_dir
MODEL_SIZE=8
LOG_PATH=logs/${MODEL_SIZE}b/$DIR
DATASET=ablation
cd llama-recipes

[ -d $LOG_PATH ] || mkdir -p $LOG_PATH

torchrun > ${LOG_PATH}/epoch1.txt \
    --nnodes 1 --nproc_per_node 4 recipes/finetuning/finetuning.py \
    --enable_fsdp \
    --use_peft --peft_method lora \
    --model_name ../llama-3.1-${MODEL_SIZE}b/${MODEL_SIZE}B \
    --fsdp_config.pure_bf16 \
    --batch_size_training 4 \
    --gradient_accumulation_steps 3 \
    --dataset custom_dataset \
    --custom_dataset.file "preprocess_data/$DATASET/chatbot_dataset.py:get_preprocessed_custom" \
    --output_dir ../llama-3.1-${MODEL_SIZE}b/chatbot\(finetuned_17k_$DIR\)/epoch1 \
    --num_epochs 1 \
    --save_model 

# continue training from last epoch
for i in `seq 1 19`
do
    # network-chatbot/llama-recipes
    torchrun > $LOG_PATH/epoch$(($i+1)).txt \
        --nnodes 1 --nproc_per_node 4 recipes/finetuning/finetuning.py \
        --enable_fsdp --low_cpu_fsdp \
        --use_peft --peft_method lora \
        --model_name ../llama-3.1-${MODEL_SIZE}b/${MODEL_SIZE}B \
        --fsdp_config.pure_bf16 \
        --batch_size_training 4 \
        --gradient_accumulation_steps 3 \
        --dataset custom_dataset \
        --custom_dataset.file "preprocess_data/$DATASET/chatbot_dataset.py:get_preprocessed_custom" \
        --output_dir ../llama-3.1-${MODEL_SIZE}b/chatbot\(finetuned_17k_$DIR\)/epoch$(($i+1)) \
        --from_peft_checkpoint ../llama-3.1-${MODEL_SIZE}b/chatbot\(finetuned_17k_$DIR\)/epoch$i \
        --num_epochs 1 \
        --save_model 
done
