# stable-diffusion-pytorch

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kjsman/stable-diffusion-pytorch/blob/main/demo.ipynb)

PyTorch implementation of [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release).

Features are pruned if not needed in Stable Diffusion (e.g. Attention mask at CLIP tokenizer/encoder). Configs are hard-coded (based on Stable Diffusion v1.x). Loops are unrolled when that shape makes more sense.

## Dependencies

* PyTorch
* Numpy
* Pillow
* regex
* tqdm

Text-to-image generation:
```py
from stable_diffusion_pytorch import pipeline

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts)
images[0].save('output.jpg')
```

...with multiple prompts:
```
prompts = [
    "a photograph of an astronaut riding a horse",
    ""]
images = pipeline.generate(prompts)
```

...with unconditional(negative) prompts:
```py
prompts = ["a photograph of an astronaut riding a horse"]
uncond_prompts = ["low quality"]
images = pipeline.generate(prompts, uncond_prompts)
```

...with seed:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, uncond_prompts, seed=42)
```

Preload models (you will need enough VRAM):
```py
from stable_diffusion_pytorch import model_loader
models = model_loader.preload_models('cuda')

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, models=models)
```

If you get OOM with above code but have enough RAM (not VRAM), you can move models to GPU when needed
and move back to CPU when not needed:
```py
from stable_diffusion_pytorch import model_loader
models = model_loader.preload_models('cpu')

prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, models=models, device='cuda', idle_device='cpu')
```

Image-to-image generation:
```py
from PIL import Image

prompts = ["a photograph of an astronaut riding a horse"]
input_images = [Image.open('space.jpg')]
images = pipeline.generate(prompts, input_images=images)
```

...with custom strength:
```py
prompts = ["a photograph of an astronaut riding a horse"]
input_images = [Image.open('space.jpg')]
images = pipeline.generate(prompts, input_images=images, strength=0.6)
```

Change [classifier-free guidance](https://arxiv.org/abs/2207.12598) scale:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, cfg_scale=11)
```

...or disable classifier-free guidance:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, do_cfg=False)
```

Reduce steps (faster generation, lower quality):
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, n_inference_steps=28)
```

Use different sampler:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, sampler="k_euler")
# "k_lms" (default), "k_euler", or "k_euler_ancestral" is available
```

Generate image with custom size:
```py
prompts = ["a photograph of an astronaut riding a horse"]
images = pipeline.generate(prompts, height=512, width=768)
```

