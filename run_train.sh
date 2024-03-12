python -m train.train_mdm_motion_control \
    --save_dir save/humanml_first_key \
    --dataset humanml \
    --inpainting_mask key \
    --resume_checkpoint ./save/humanml_first_key/model.pt
    # --only_text_condition \
