python run_language_modeling.py \
--output_dir=OUTPUT_PATH \
--model_type=gpt2 \
--model_name_or_path=gpt2 \
--do_train \
--line_by_line \
--train_data_file=DATA.txt \
--overwrite_output_dir \
--per_gpu_train_batch_size 2 \
--num_train_epochs 1
