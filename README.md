# Medical Image Generation in Spain

Synthetic grayscale image generation with a vanilla GAN and pretrained MediGAN
models, with a Streamlit interface.

## Data and models

The vanilla GAN expects authorized medical training images under a directory,
including the course layout `data_p5/gan_training_images/`. Images are converted
to grayscale, resized to 64 × 64, and normalized to [−1, 1]. No patient images or
model weights are bundled.

The separate MediGAN workflow uses published generators for modalities such as
mammography and chest X-ray. Each model has its own source dataset, weights,
dependencies, and license; see the [MediGAN model documentation](https://medigan.readthedocs.io/en/latest/).

## Setup and usage

Use Python 3.11 and a separate virtual environment.

```bash
python -m pip install -r requirements.txt
python train.py --data data_p5/gan_training_images
python generate.py --checkpoint outputs/gan/gan.pt --count 4 --output outputs/samples
streamlit run app.py
```

To use a pretrained MediGAN model:

```bash
python generate.py --medigan-model 00001_DCGAN_MMG_CALC_ROI --count 4 --output outputs/medigan --install-dependencies
```

Model-specific dependencies and weights are obtained on demand. Install them in
this project's environment; some upstream generators may require additional
platform-specific setup.

The vanilla GAN uses an MLP generator with a Tanh output and a discriminator
trained with binary cross-entropy logits. Training alternates discriminator and
generator updates using Adam. It saves `gan.pt`, `history.csv`, and a sample grid.
The app supports both sources and image downloads.

## Evaluation

Generator/discriminator losses and sample grids are training diagnostics, not
proof of medical realism, privacy preservation, or clinical usefulness. No FID,
external validation score, or clinical evaluation is claimed. Synthetic samples
are for educational image-synthesis experiments.

## References and license

[WorldQuant University](https://www.wqu.edu/) provided the original course material.
[MediGAN](https://github.com/RichardObi/medigan) provides pretrained model integration.
Standalone scripts and documentation use MIT; course materials, medical datasets,
and pretrained models retain their own licenses and terms.
