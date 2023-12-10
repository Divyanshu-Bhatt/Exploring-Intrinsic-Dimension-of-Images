latent_dim=128
save_dir=samples/basenji_${latent_dim}


python generate_data/gen_images.py \
  --num_samples 5000 \
  --class_name basenji \
  --latent_dim $latent_dim \
  --batch_size 100 \
  --save_dir $save_dir 