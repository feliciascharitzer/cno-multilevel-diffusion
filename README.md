# Convolutional Neural Operators for Score-Based Diffusion Models

This repository implements Convolutional Neural Operators (CNOs) to learn the score operator of infinite-dimensional Score-Based Diffusion Models. The CNO framework can be tested against Fourier Neural Operator (FNO) and UNet architectures to evaluate its ability to generate images at arbitrary resolutions.

## Run code
The model can be trained with `python main.py`. Hyperparameters like the following values can additionally be specified:
```bash
python main.py [--model {'unet', 'fno', 'cno'}] [--prior_name {'standard', 'lap_conv', 'fno', 'combined_conv'}] [--seed SEED] [--subset {0, 1}]
```
Setting `subset` to 1 trains the model on a reduced set of MNIST images to reduce training time.

To test a trained model, run:
```bash
python test.py --save_model 'checkpoint_name' [--upscale SCALING_FACTOR] [--input_height INPUT_HEIGHT]
```
The parameter `upscale` represents the factor by which the training resolution is increased in the evaluation of the model, e.g. training the model at an input height of 28 and evaluating it at height 56 should be indicated by `--upscale 2`.

## Results

Training was performed on the MNIST dataset at resolution 28x28. The models were then evaluated at resolutions 28x28 and 56x56 using the MMD score. Results show consistent performance of the CNO architecture across resolutions and improved stability of the CNO compared to FNO and UNet when evaluated on the Laplacian prior.

MMD values for the Standard prior:
| Model  | UNet            | FNO             | CNO             |
|--------|-----------------|-----------------|-----------------|
| MMD 28 | 0.0311 ± 0.0197 | 0.0446 ± 0.0505 | 0.0147 ± 0.0032 |
| MMD 56 | 1.9605 ± 0.0396 | 1.1963 ± 0.2289 | 0.0448 ± 0.0074 |


MMD values for the Laplacian prior:
| Model  | UNet            | FNO             | CNO             |
|--------|-----------------|-----------------|-----------------|
| MMD 28 | 2.1833 ± 1.7403 | 2.3430 ± 1.8237 | 0.0658 ± 0.0521 |
| MMD 56 | 3.4154 ± 2.1862 | 4.1955 ± 2.4331 | 0.1048 ± 0.0470 |

## References

This code extends the framework in [multilevelDiff](https://github.com/PaulLyonel/multilevelDiff) to add CNOs as an architecture for score learning using the CNO implementation in [ConvolutionalNeuralOperator](https://github.com/camlab-ethz/ConvolutionalNeuralOperator).

The implementation is based on the following papers:
1. [Multilevel Diffusion: Infinite Dimensional Score-Based Diffusion Models for Image Generation](https://arxiv.org/abs/2303.04772v3) (Hagemann et al., 2023)
2. [Convolutional Neural Operators for robust and accurate learning of PDEs](https://arxiv.org/abs/2310.15017) (Raonić et al., 2023)
