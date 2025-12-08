# generative-photography

This project is derived from [[CVPR 2025 Highlight] Generative Photography](https://arxiv.org/abs/2412.02168).

> **Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis** <br>
> [Yu Yuan](https://yuyuan-space.github.io/), [Xijun Wang](https://www.linkedin.com/in/xijun-wang-747475208/), [Yichen Sheng](https://shengcn.github.io/), [Prateek Chennuri](https://www.linkedin.com/in/prateek-chennuri-3a25a8171/), [Xingguang Zhang](https://xg416.github.io/), [Stanley Chan](https://engineering.purdue.edu/ChanGroup/stanleychan.html)<br>

The original github page is https://github.com/pandayuanyu/generative-photography

## [[Paper](https://arxiv.org/abs/2412.02168)] [[Project Page](https://yuyuan-space.github.io/GenerativePhotography/)] [[Dataset](https://huggingface.co/datasets/pandaphd/camera_settings)] [[Weights](https://huggingface.co/pandaphd/generative_photography)] [[HF Demo](https://huggingface.co/spaces/pandaphd/generative_photography)]


## Configurations

Please see the Configurations in author's [github](https://github.com/pandayuanyu/generative-photography) page.

## Inference

```python 
# For bokeh rendering
python inference_bokehK.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml --base_scene "A young boy wearing an orange jacket is standing on a crosswalk, waiting to cross the street." --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]"

# For focal length
python inference_focal_length.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml --base_scene "A cozy living room with a large, comfy sofa and a coffee table." --focal_length_list "[25.0, 35.0, 45.0, 55.0, 65.0]"

# For shutter speed
python inference_shutter_speed.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_shutter_speed.yaml --base_scene "A modern bathroom with a mirror and soft lighting." --shutter_speed_list "[0.1, 0.3, 0.52, 0.7, 0.8]"

# For color temperature 
python inference_color_temperature.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_color_temperature.yaml --base_scene "A blue sky with mountains." --color_temperature_list "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]"
```
