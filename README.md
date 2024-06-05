#  💫 STAResNet 💫: a Network in Spacetime Algebra to solve Maxwell's PDEs 

<p align="center">
<img src="/figures/3Dslices.gif" width="700">
</p>

Presented at the Applied Geometric Algebras in Computer Science and Engineering Conference (AGACSE) 2024 in Amsterdam, Netherlands.

## Intro 💫

<p align="center">
<img src="/figures/3D.gif" width="700">
</p>

A same set of PDEs can be expressed in different algebras. We focus on Maxwell's PDEs, which can be formulated in 2D and 3D GA (for the 2D and 3D case), but also in 3D and 4D 
**Spacetime Algebra (STA)**. It is known in the literature that a STA formulation of Maxwell's PDEs is much more compact and elegant, as well as easier to compute (Lasenby, 2020). We demonstrated how this holds true also for **Clifford Algebra Networks** that work with STA multivectors as opposed to vanilla GA. 

STAResNet is our own ResNet-like Clifford Algebra Network that represents inputs, weights and biases as STA multivectors. We showed how solving Maxwell's PDEs via our STAResNet outperforms Clifford ResNet (Brandstetter et al., 2022) which works in 3D GA.




## Achievements 💫

- 🌟 Extended experiments of Brandstetter et al. on Maxwell's PDEs to the 2D case
- 🌟 Implemented 3D convolutional layers for *any* algebra of *any* dimension in TensorFlow
- 🌟 Implemented a network working *exclusively* with objects in Spacetime Algebra
- 🌟 Demonstrated how STAResNet solves PDEs more accurately than Clifford ResNet both in 2D and 3D space
- 🌟 Showed how STAResNet is more resilient in presence of obstacles, either seen or unseen
- 🌟 Achieved up to 2.6 lower MSE error between GT and esitmated fields with 6 times fewer trainable parameters as opposed to 

We can conclude that choice of the algebra, when implementing Clifford Networks, is key for more lightweight, descriptive and accurate networks that are grounded in the physics of the problem and represent a more natural parametrisation of the chosen problem.


## Requirements 💫

<p align="center">
<img src="/figures/2D2.gif" width="700">
</p>


STAResNet requires the following:

- tensorflow 2.5.0
- fdtd
- tfga
- clifford

## How to run 💫

```
python datagen3D.py
```

```
python GAmaxwell3D.py
```






