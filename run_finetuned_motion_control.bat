python finetuned_motion_control.py ^
    --model_path save/humanml_only_text_condition/model.pt ^
    --dataset humanml ^
    --guidance_param 2.5 ^
    --num_samples 30 ^
    --inpainting_mask only_text ^
    --text_condition "do sit-ups on a bed." 


@REM python finetuned_motion_control_posa.py ^
@REM     --model_path save/humanml_root_key/model.pt ^
@REM     --dataset humanml ^
@REM     --guidance_param 2.5 ^
@REM     --num_samples 30 ^
@REM     --show_input ^
@REM     --text_condition "a man stands up from the floor, walks. and sits on a chair."


@REM python finetuned_motion_control_posa_trajectory.py ^
@REM     --model_path save/humanml_root_key/model.pt ^
@REM     --dataset humanml ^
@REM     --guidance_param 2.5 ^
@REM     --num_samples 30 ^
@REM     --show_input ^
@REM     --text_condition "walk to a sofa and sits on it"