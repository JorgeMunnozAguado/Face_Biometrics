
conda create -y --name faceBio python=3.6

conda activate faceBio

conda install -y keras-gpu tensorflow-gpu
conda install numpy pillow
pip install opencv-python keras-vggface Keras-Applications
pip install pandas scikit-learn matplotlib


https://dauam-my.sharepoint.com/:u:/g/personal/aythami_morales_uam_es/ERd0YZG26FlGl1hr9nQtd54BNmW2XMwuzS-LXh0DoMp2ig?e=f8jD7w


Etiquetas:   HA, HB, HN, MA, MB, MN

H : Hombre
M : Mujer

A : Asiático
B : Blanco
N : Negro




# Resultados

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

# Grad-cam

https://keras.io/examples/vision/grad_cam/
