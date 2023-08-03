# CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D Scene Layout
<a href="https://arxiv.org/abs/2303.13843"><img src="https://img.shields.io/badge/arXiv-2303.13843-b31b1b.svg" height=22.5></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" height=22.5></a>  

> Recent research endeavors have shown that combining neural radiance fields (NeRFs) with pre-trained diffusion models holds great potential for text-to-3D generation.
> However, a hurdle is that they often encounter guidance collapse when rendering complex scenes from multi-object texts. 
> Because the text-to-image diffusion models are inherently unconstrained, making them less competent to accurately associate object semantics with specific 3D structures. 
> To address this issue, we propose a novel framework, dubbed CompoNeRF, that explicitly incorporates an editable 3D scene layout to provide effective guidance at the single object (i.e., local) and whole scene (i.e., global) levels. 
> Firstly, we interpret the multi-object text as an editable 3D scene layout containing multiple local NeRFs associated with the object-specific 3D box coordinates and text prompt, which can be easily collected from users. 
> Then, we introduce a global MLP to calibrate the compositional latent features from local NeRFs, which surprisingly improves the view consistency across different local NeRFs. 
> Lastly, we apply the text guidance on global and local levels through their corresponding views to avoid guidance ambiguity. 
> This way, our CompoNeRF allows for flexible scene editing and re-composition of trained local NeRFs into a new scene by manipulating the 3D layout or text prompt. 
> Leveraging the open-source Stable Diffusion model, our CompoNeRF can generate faithful and editable text-to-3D results while opening a potential direction for text-guided multi-object composition via the editable 3D scene layout.

## Description :scroll:	
Official Implementation for "CompoNeRF: Text-guided Multi-object Compositional NeRF with Editable 3D Scene Layout".

> TL;DR - We explore different ways of introducing shape-guidance for Text-to-3D and present three models: a purely text-guided Latent-NeRF, Latent-NeRF with soft shape guidance for more exact control over the generated shape, and Latent-Paint for texture generation for explicit shapes.



## Recent Updates :newspaper:
* `13.07.2023` - Code release

* `14.03.20223` - Created initial repo


### CompoNeRF
Here we apply a text-to-3D from the multi-object text. 

Similar to Latent-NeRF, we directly train each local NeRF in latent space.


To create such results, run the `train_compo_nerf` script. Parameters are handled using [pyrallis](https://eladrich.github.io/pyrallis/) and can be passed from a config file or the cmd.

```bash
 python -m scripts.train_compo_nerf --config_path demo_configs/compo_nerf/apple_and_banana.yaml
```



## Getting Started

```
conda create -n componerf python=3.9 -y
conda activate componerf
python -m pip install --upgrade pip
```

### Installation :floppy_disk:	
Please refer to [Pytorch](https://pytorch.org/) for customized installation. 
Install the common dependencies from the `requirements.txt` file
```bash
pip install -r requirements.txt
```

For more details about installation and pre-trained diffusion models, please refer to the installation of [Latent-NeRF](https://github.com/eladrich/latent-nerf#installation-floppy_disk)
_

### Training :weight_lifting:	

Scripts for training are available in the `scripts/` folder, see above or in the `demo_configs/` for some actual examples. 
```
export CONFIG_FILE=demo_configs/compo_nerf/tabel_wine.yaml
export GPUs=0

CUDA_VISIBLE_DEVICES=$GPUs,
python train.py \
    --config_path $CONFIG_FILE
```

Note that you also need a :hugs: token for StableDiffusion. First accept conditions for the model you want to use, default one is [`CompVis/stable-diffusion-v1-4`]( https://huggingface.co/CompVis/stable-diffusion-v1-4). Then, add a TOKEN file(./TOKEN) [access token](https://huggingface.co/settings/tokens) to the root folder of this project, or use the `huggingface-cli login` command

If error ("HTTPSConnectionPool(host='huggingface.co', port=443)), please refer to the [issue](https://github.com/huggingface/transformers/issues/17611#issuecomment-1323272726).

```
pip install --upgrade requests==2.27.1
```
### Additional Tips and Tricks:	

* Check out the `vis/train` to see the actual rendering used during the optimization.


## Acknowledgments
The `CompoNeRF` code is heavily based on the [Latent-NeRF](https://github.com/eladrich/latent-nerf) project, and some codes borrows from [NeuralSceneGraph](https://github.com/princeton-computational-imaging/neural-scene-graphs).

## Citation
If you use this code for your research, please cite our paper [Componerf: Text-guided multi-object compositional nerf with editable 3d scene layout](https://arxiv.org/abs/2303.13843)

```
@article{lin2023componerf,
  title={Componerf: Text-guided multi-object compositional nerf with editable 3d scene layout},
  author={Lin, Yiqi and Bai, Haotian and Li, Sijia and Lu, Haonan and Lin, Xiaodong and Xiong, Hui and Wang, Lin},
  journal={arXiv preprint arXiv:2303.13843},
  year={2023}
}
```
