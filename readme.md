After making a CNN with nn.functional, i decided to finally stop playing around with those tools and finally learn how to do nn.module, and why not start with a CNN for CIFAR-10?

I designed this custom Convolutional Neural Network (CNN) from scratch to make the definitive leap from simple MLPs (such as MNIST) to processing real-world color images in CIFAR-10. I wanted to experiment with modular convolutional block design, understand how to regularize deep networks, and see firsthand the difference in spatial and semantic complexity that a convolutional architecture introduces.

## Key Learnings:

### Training

The leap from MNIST to CIFAR-10 is monumental in terms of variance. To prevent the network from memorizing the images, I learned to implement advanced data augmentation techniques on the $32 \times 32$ color images:
* **Random Crop & Horizontal Flip:** To teach the network translation and orientation invariance.
* **Random Erasing (`p=0.3`):** I learned that covering random patches of the image forces the network not to depend on a single local feature (like an eye or a wheel) to recognize the entire object, forcing it to generalize.
* **Label Smoothing (`0.1`):** I understood that pure cross-entropy can make the network excessively overconfident in its predictions. Smoothing the labels trains the model with a healthy level of uncertainty.

### Mathematics

Convolutions act as spatial filters that learn to extract information locally and hierarchically (edges -> textures -> complex shapes).
* **Batch Normalization:** I experimented with how normalizing activations after each convolution stabilizes the mean and variance during training. This significantly accelerates convergence and enables training much deeper networks (10 convolutional layers in this case) without gradient vanishing or explosion.
* **Weight Decay (`1e-4`):** I understood how L2 regularization penalizes excessively large weights in the Adam optimizer to maintain the smoothness of the model's function.

### Architecture

I designed a highly flexible modular pipeline parameterized by lists:
* **Continuous Convolutional Blocks:** I stacked 10 convolutional layers (`Conv_Blocks = [8,8,16,16,32,32,64,64,128,128]`) strategically interleaved with `MaxPool2d` and `Dropout2d` to gradually reduce spatial resolution (from 32x32 to 2x2) while increasing channel depth (from 3 to 128).
* **Global Average Pooling (GAP):** This is one of my greatest learnings. Instead of flattening the entire tensor at the end (which would create a massive parameter bottleneck of millions of weights in a dense layer, causing severe overfitting), I used `nn.AdaptiveAvgPool2d((1,1))` to collapse the channels and project them directly with a single linear layer to the 10 output classes. This completely destroys translation equivariance, making it the perfect translation-invariant output for what an MLP classifier requires.

### Code

* **Data Loading Strategies:** I wrote the code to support two modes:
  1. Standard use of `DataLoader` with parallel subprocesses (`num_workers=2`).
  2. Loading the entire test set directly into VRAM, leveraging the fact that the dataset easily fits on the GPU to completely eliminate I/O transfer bottlenecks at each epoch.
* **Dynamic Modularity:** I used Python's flexibility to dynamically generate the convolutional architecture by iterating over configuration lists using `nn.Sequential(*self.layers)`. This allows me to change the number of filters or layers in seconds simply by modifying the `conv_blocks` list.

---

## 4. Baseline Experiment Results (30 Epochs)

To verify the convergence and performance of this `nn.Module`-based modular CNN, a baseline experiment of **30 epochs** was executed locally on CPU:
* **Filters Configuration:** $[8, 8, 16, 16, 32, 32, 64, 64, 128, 128]$
* **Batch Size:** $128$
* **Initial Learning Rate:** $0.0015$ (Adam)
* **Weight Decay:** $1\times10^{-4}$

### Training Logs:

```text
Epoch 1/30 | Loss: 1.9545 | Train Acc: 30.64% | Test Acc: 43.23% | Time: 17.7s
Epoch 5/30 | Loss: 1.5860 | Train Acc: 49.61% | Test Acc: 60.74% | Time: 15.7s
Epoch 10/30 | Loss: 1.4406 | Train Acc: 56.96% | Test Acc: 66.02% | Time: 15.7s
Epoch 15/30 | Loss: 1.3745 | Train Acc: 60.36% | Test Acc: 67.81% | Time: 15.7s
Epoch 20/30 | Loss: 1.3325 | Train Acc: 62.42% | Test Acc: 71.37% | Time: 15.7s
Epoch 25/30 | Loss: 1.2974 | Train Acc: 64.03% | Test Acc: 72.60% | Time: 15.8s
Epoch 30/30 | Loss: 1.2750 | Train Acc: 65.30% | Test Acc: 73.96% | Time: 16.7s
```

### Architectural Analysis:
* **Excellent Generalization:** The model finishes training at **73.96% test accuracy** while maintaining a training accuracy of **65.30%**. The test accuracy is significantly higher than the training accuracy because the powerful training regularizations (such as `RandomErasing` with $p=0.3$, `Dropout2d` with $p=0.2$, and `Label Smoothing` of $0.1$) are active *only* during the training phase. When the model is evaluated (`model.eval()`), these regularizers are disabled, allowing the network to make cleaner, highly accurate predictions.
* **Stable Convergence:** The training curve converges exceptionally smoothly. Even starting with only 8 filters in the first blocks, stacking 10 convolutional layers with `nn.BatchNorm2d` after every single layer prevents gradient problems entirely and guides the network to stable representations.
