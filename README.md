
#  💫 STAResNet 


<p align="center">
<img src="/figures/3Dslices.gif" width="900">
</p>


Presented with the title "STAResNet: a Network in Spacetime Algebra to solve Maxwell's PDEs" at the 9th  **Applied Geometric Algebras in Computer Science and Engineering Conference (AGACSE)** in Amsterdam, Netherlands, in August 2024.


## Background 💫

A same set of PDEs can be expressed in different algebras. We focus on Maxwell's PDEs, which can be formulated in vanilla GA but also in **Spacetime Algebra (STA)**. An STA formulation of Maxwell's PDEs is not only more compact and elegant as opposed to vanilla GA, but also easier to compute (Lasenby, 2020). We demonstrated how this holds true also when PDEs are computed through learned methods, such as **Clifford Algebra Networks**, when they operate with STA multivectors as opposed to vanilla GA ones. 

💫 **STAResNet** 💫 is our own ResNet-like Clifford Algebra Network that represents inputs, weights and biases as STA multivectors. We showed how solving Maxwell's PDEs via STAResNet outperforms Clifford ResNet (Brandstetter et al., 2022), which works in vanilla GA instead.




## Achievements 💫

- Extended experiments of Brandstetter et al. on Maxwell's PDEs to the 2D case
- Implemented 3D convolutional layers for *any* algebra of *any* dimension in TensorFlow
- Implemented a network working *exclusively* with objects in Spacetime Algebra
- Demonstrated how STAResNet solves PDEs more accurately than Clifford ResNet both in 2D and 3D space
- Showed how STAResNet is more resilient in presence of obstacles, either previously seen or unseen
- Achieved up to 2.6 lower MSE error between GT and esitmated fields with 6 times fewer trainable parameters as opposed to 2D Clifford ResNet

We can conclude that the choice of the correct algebra, when implementing Clifford Networks, is key for more lightweight, descriptive and accurate networks that are grounded in physics and represent a more natural parametrisation for the chosen problem.

<p align="center">
<img src="/figures/3D.gif" width="600">
</p>


## Requirements 💫

STAResNet requires the following:

- tensorflow 2.5.0
- fdtd
- tfga
- clifford

<p align="center">
<img src="/figures/2D2.gif" width="600">
</p>

## How to run 💫

3 steps required. The example below is for STAResNet for 3D EM fields, but other combinations follow the same approach.

1. generate data:
```
python datagen3D.py
```
2. train:
```
python maxwell_STA_3D.py
```
3. test:
```
python test_STA_3D.py
```

## How to cite 💫

If you are referencing **STAResNet** in your work, please cite the following:

```bibtex
@article{pepe2024staresnet,
  title={STAResNet: a Network in Spacetime Algebra to solve Maxwell's PDEs},
  author={Pepe, Alberto and Buchholz, Sven and Lasenby, Joan},
  journal={arXiv preprint arXiv:2408.13619},
  year={2024}
}
```

## Other 💫

If you want to experiement with STAResNet without retraining it, most pretrained models and datasets can be requested from the authors via ap2219 [at] cam [dot] ac [dot] uk 👨🏼‍🎤👩🏾‍🎤

## Acknowledgements 💫

The original scripts `pde.py` and `datagen.py` have been written by Brandstetter et al. (2022). The script `layers.py` has been expanded from that of Christian Hockey. 
