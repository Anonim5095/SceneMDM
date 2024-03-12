# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append("/diffusion")
sys.path.append("/")

from diffusion.inpainting_gaussian_diffusion import InpaintingGaussianDiffusion
from diffusion.respace import SpacedDiffusion
from model.model_blending import ModelBlender
from utils.fixseed import fixseed
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.parser_util import edit_inpainting_args
from utils.model_util import load_model_blending_and_diffusion
from utils import dist_util
from model.cfg_sampler import wrap_model
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml_utils import get_inpainting_mask
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion, plot_3d_pose
import shutil

def main():

    # posa_output_path = "../POSA/save/debug_Werkraum_default/pkl/Werkraum/rp_ellie_posed_010_0_0_00.npy" # put to shelf
    posa_output_path = "./save/humanml_only_text_condition/result_a_person_is_wiping_a_table/sample01_rep00_iter=20_new_opt_cam_t_affordance/pkl/MPH11/057.npy"

    args_list = edit_inpainting_args()
    args = args_list[0]
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml', 'humanml_smplx'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    
    # load posa data
    posa_data = np.load(posa_output_path, allow_pickle=True).item()
    posa_pose = posa_data['data']
    motion_length = posa_data['length']
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    n_joints = 22

    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                '{}'.format(posa_data['scene_name']))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')[:40]

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size,
                              num_frames=max_frames,
                              split='test',
                              load_mode='train',
                              size=args.num_samples)  # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    DiffusionClass = InpaintingGaussianDiffusion if args_list[0].filter_noise else SpacedDiffusion
    model, diffusion = load_model_blending_and_diffusion(args_list, data, dist_util.dev(), DiffusionClass=DiffusionClass)

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    if args.text_condition != '':
        texts = [args.text_condition] * args.num_samples
        model_kwargs['y']['text'] = texts
        

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions
    model_kwargs['y']['inpainting_mask'] = \
        torch.tensor(get_inpainting_mask(args.inpainting_mask, 
                                         input_motions.shape, 
                                         model_kwargs['y']['lengths'], 
                                         only_text=args.only_text_condition, 
                                         keyframes=posa_data['keyframes']
                                         )).float().to(dist_util.dev())


    # # POSAのOutput Poseを確認
    # posa_pose = posa_pose.reshape(1, 1, -1, 263)
    # motion_ = recover_from_ric(torch.Tensor(posa_pose), n_joints)
    # motion_ = motion_.view(-1, *motion_.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    # save_path_ = "./save/rp_alexandra_posed_001_0_0_00_.mp4"
    # plot_3d_motion(save_path_, skeleton, motion_, title="",
    #         dataset=args.dataset, fps=fps, vis_mode='gt',
    #         gt_frames=gt_frames_per_sample.get(0, []))


    # 入力モーションをkeypose以外平均値に
    mean_motion = torch.Tensor(data.dataset.mean).unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(args.num_samples, 1, 196, 1).permute(0, 3, 1, 2).to(args.device)
    mean_motion *= (1. - model_kwargs['y']['inpainting_mask'])
    input_motions = input_motions * model_kwargs['y']['inpainting_mask'] + mean_motion
    
    posa_pose = (posa_pose - data.dataset.mean) / data.dataset.std

    # # 20フレームずつズラしてkeyposeを挟み込んだ入力モーションとマスクを作成
    # input_motions = torch.zeros([10, 263, 1, 196])
    # input_masks = torch.zeros([10, 263, 1, 196])
    # for i in range(len(input_motions)):
    #     inmotion = torch.zeros([196, 263])
    #     mask = torch.zeros([196, 263])
    #     for frame in range(len(inmotion)):
    #         mask[frame][:3] = 1
    #         inmotion[frame] = torch.Tensor(mean)
    #         if frame == i * 20:
    #             mask[frame] = 1
    #             inmotion[frame] = torch.Tensor(posa_pose[0])
    #     input_motions[i] = inmotion.T.unsqueeze(1)
    #     input_masks[i] = mask.T.unsqueeze(1)
    # input_motions = input_motions.to(dist_util.dev())
    # model_kwargs['y']['inpainted_motion'] = input_motions
    # model_kwargs['y']['inpainting_mask'] = input_masks.float().to(dist_util.dev())


    all_motions = []
    all_lengths = []
    all_text = []
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        mask = model_kwargs['y']['inpainting_mask'].detach().cpu().numpy()

        keyframes = []
        for i in range(len(mask)):
            a = mask[i, :, 0, :].T
            plt.imshow(a, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
            plt.title('Mask Visualization')
            plt.colorbar()
            save_dir = out_path + "/masks/"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir + "{}_mask_key.png".format(i))
            plt.close()

            tmp = []
            for frame in range(len(a)):
                if sum(a[frame] == 0) == 0:
                    tmp.append(frame)
            keyframes.append(tmp)
        
        if not args.only_text_condition:
            # KeyPoseの差し込み
            for i in range(len(keyframes)):
                if len(keyframes[i]) == 0:
                    continue
                k = keyframes[i][-1]
                input_motions[i][:, 0, k][3:] = torch.Tensor(posa_pose[0][3:])
                # input_motions[i][:, 0, k][0] += np.pi / 2

        input_motions = input_motions.to(dist_util.dev())
        model_kwargs['y']['inpainted_motion'] = input_motions
        model_kwargs['y']['lengths'] = torch.zeros(model_kwargs['y']['lengths'].shape) + motion_length

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=input_motions,
            progress=True,
            dump_steps=None,
            noise=None, 
            const_noise=False,
        )


        # Recover XYZ *positions* from HumanML3D vector representation
        if model.data_rep == 'hml_vec':
            n_joints = 22 if sample.shape[1] == 263 else 21
            sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
            sample = recover_from_ric(sample, n_joints)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        all_text += model_kwargs['y']['text']
        all_motions.append(sample.cpu().numpy())
        all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy().astype(np.int64))

        print(f"created {len(all_motions) * args.batch_size} samples")


    all_motions = np.concatenate(all_motions, axis=0)
    all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    all_text = all_text[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)

    npy_path = os.path.join(out_path, 'results.npy')
    print(f"saving results file to [{npy_path}]")
    np.save(npy_path,
            {'motion': all_motions, 'text': all_text, 'lengths': all_lengths, 
             'keyframes': keyframes, 't_fm_orig': posa_data['t_fm_orig'], 'R_fm_orig': posa_data['R_fm_orig'], 
             'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
        fw.write('\n'.join(all_text))
    with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
        fw.write('\n'.join([str(l) for l in all_lengths]))

    print(f"saving visualizations to [{out_path}]...")
    skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

    # Recover XYZ *positions* from HumanML3D vector representation
    if model.data_rep == 'hml_vec':
        input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
        input_motions = recover_from_ric(input_motions, n_joints)
        input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()


    for sample_i in range(args.num_samples):
        rep_files = []
        if args.show_input:
            caption = 'Input Motion'
            length = model_kwargs['y']['lengths'][sample_i]
            motion = input_motions[sample_i].transpose(2, 0, 1)[:int(length)]
            keyframe = keyframes[sample_i]
            save_file = 'input_motion{:02d}.mp4'.format(sample_i)
            if os.path.exists(save_file):
                continue
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | -> {save_file}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title="",
                        dataset=args.dataset, fps=fps, vis_mode='gt',
                        gt_frames=gt_frames_per_sample.get(sample_i, []),
                        keyframes=keyframe)
        for rep_i in range(args.num_repetitions):
            caption = all_text[rep_i*args.batch_size + sample_i]
            if args.guidance_param == 0:
                caption = 'Edit [{}] unconditioned'.format(args.inpainting_mask)
            else:
                caption = 'Edit [{}]: {}'.format(args.inpainting_mask, caption)
            length = all_lengths[rep_i*args.batch_size + sample_i]
            motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:int(length)]
            save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
            animation_save_path = os.path.join(out_path, save_file)
            rep_files.append(animation_save_path)
            print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {animation_save_path}]')
            plot_3d_motion(animation_save_path, skeleton, motion, title="",
                        dataset=args.dataset, fps=fps, vis_mode=args.inpainting_mask,
                        gt_frames=gt_frames_per_sample.get(sample_i, []), painting_features=args.inpainting_mask.split(','),
                        keyframes=keyframe)
            # Credit for visualization: https://github.com/EricGuo5513/text-to-motion

        if args.num_repetitions > 1:
            all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
            ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
            hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions + (1 if args.show_input else 0)} '
            ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
            os.system(ffmpeg_rep_cmd)
            print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')

    abs_path = os.path.abspath(out_path)
    print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()