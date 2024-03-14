# SceneMDM: Harnessing Scene Interactions with Affordances for Text-to-Motion

<!-- [![arXiv](https://img.shields.io/badge/arXiv-<2303.01418>-<COLOR>.svg)](https://arxiv.org/abs/2303.01418) -->

The official PyTorch implementation of the paper [**"Harnessing Scene Interactions with Affordances for Text-to-Motion"**]().

<!-- Please visit our [**webpage**](https://priormdm.github.io/priorMDM-page/) for more details. -->

<img src="./images/results.gif" width="800">

<!-- #### Bibtex
If you find this code useful in your research, please cite:

```
@article{shafir2023human,
  title={Human motion diffusion as a generative prior},
  author={Shafir, Yonatan and Tevet, Guy and Kapon, Roy and Bermano, Amit H},
  journal={arXiv preprint arXiv:2303.01418},
  year={2023}
}
``` -->


## Getting started

This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.9
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment 

Install ffmpeg (if not already installed):

```shell
sudo apt update
sudo apt install ffmpeg
```
For windows use [this](https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/) instead.

Setup conda env:
```shell
conda env create -f environment.yml
conda activate PriorMDM
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/GuyTevet/smplx.git
```

### 2. Get MDM dependencies

PriorMDM share most of its dependencies with the original MDM. 
If you already have an installed MDM from the official repo, you can save time and link the dependencies instead of getting them from scratch.

<details>
  <summary><b>If you already have an installed MDM</b></summary>

**Link from installed MDM**

Before running the following bash script, first change the path to the full path to your installed MDM

```bash
bash prepare/link_mdm.sh
```

</details>


<details>
  <summary><b>First time user</b></summary>

**Download dependencies:**

```bash
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh
```

**Get HumanML3D dataset** (For all applications):

Follow the instructions in [HumanML3D](https://github.com/EricGuo5513/HumanML3D.git),
then copy the result dataset to our repository:

```shell
cp -r ../HumanML3D/HumanML3D ./dataset/HumanML3D
```

</details>

### 4. Download the pretrained models

Download the model(s) you wish to use, then unzip and place it in `./save/`.

<details>
  <summary><b>DoubleTake (long motions)</b></summary>

* [my_humanml-encoder-512](https://drive.google.com/file/d/1RCqyKfj7TLSp6VzwrKa84ldEaXmVma1a/view?usp=share_link) (This is a reproduction of MDM best model without any changes)

</details>

## Motion Synthesis 



</details>

<details>

<summary><b>Fine-tuned motion control</b></summary>

Evaluate the motion control models on the horizontal part of trajectories sampled from the test set of HumanML3D dataset.
```shell
python -m eval.eval_finetuned_motion_control --model_path save/root_horizontal_finetuned/model000280000.pt --replication_times 10
```

This code should produce a file named `eval_humanml_root_horizontal_finetuned_000280000_gscale2.5_mask_root_horizontal_wo_mm.log`, or generally:
`eval_humanml\_<model_name>\_gscale<guidance_free_scale>\_mask\_<name_of_control_features>_<evaluation_mode>.log`

</details>

## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[PriorMDM](https://github.com/priorMDM/priorMDM),
[MDM](https://github.com/GuyTevet/motion-diffusion-model),
[POSA](https://github.com/mohamedhassanmus/POSA),
[guided-diffusion](https://github.com/openai/guided-diffusion), 
[MotionCLIP](https://github.com/GuyTevet/MotionCLIP), 
[text-to-motion](https://github.com/EricGuo5513/text-to-motion), 
[actor](https://github.com/Mathux/ACTOR), 
[joints2smpl](https://github.com/wangsen1312/joints2smpl),
[TEACH](https://github.com/athn-nik/teach).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
