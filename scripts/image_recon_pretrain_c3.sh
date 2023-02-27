python train.py \
    --gpus=1 \
    --resume=/pasteur/u/esui/data/lafite/pre-trained-google-cc-best-fid.pkl \
    --outdir=/pasteur/u/esui/data/lafite/ckpt/ \
    --cfg_path=configs/img_recon.yaml \
    --temp=0.5 \
    --itd=10 \
    --itc=10 \
    --gamma=10 \
    --mirror=1 \
    --kimg=415 \
    --mixing_prob=1.0 \
    --normalize_prefix \
    --remove_mean \
    --add_gaussian_noise
    
# mixing_prob==1 -> no text data used \
# kim 415 is a little over 1 epoch of text-to-img (approx 5 epochs of img recon)