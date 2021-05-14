# Face Biometrics: Fairness and Biases


## Environment instalation

To install the environment run the following commands:

```
conda create -y --name faceBio python=3.6

conda activate faceBio

conda install -y keras-gpu==2.3.1 tensorflow-gpu==2.1.0
conda install -y numpy pillow
pip install opencv-python keras-vggface Keras-Applications
pip install pandas scikit-learn matplotlib ipython seaborn
```

Download the dataset: [Face Biometrics](https://dauam-my.sharepoint.com/:u:/g/personal/aythami_morales_uam_es/ERd0YZG26FlGl1hr9nQtd54BNmW2XMwuzS-LXh0DoMp2ig?e=f8jD7w)


## Dataset labeling

Special labeling was used to identify gender and ethnicities.

```
Labels:   HA, HB, HN, MA, MB, MN
```

Where each letter corresponds with bellow description:

```
H : Hombre (Man)
M : Mujer (Woman)
-----------------
A : Asiático (Asian)
B : Blanco (White)
N : Negro (Black)
```


## Results

By running experiments we ended with the next results:

```
Model trained with Asiaticos       accuracy (loss):
  -> test asiaticos: 0.98 (0.5152)
  -> test blancos: 0.87 (0.5656)
  -> test negros: 0.92 (0.5384)
---------------------------------------------


Model trained with Blancos       accuracy (loss):
  -> test asiaticos: 0.86 (0.4288)
  -> test blancos: 1.00 (0.2720)
  -> test negros: 0.87 (0.4153)
---------------------------------------------


Model trained with Negros       accuracy (loss):
  -> test asiaticos: 0.93 (0.4466)
  -> test blancos: 0.92 (0.4587)
  -> test negros: 0.96 (0.4279)
---------------------------------------------
```

## References

[Grad-cam](https://keras.io/examples/vision/grad_cam/)
