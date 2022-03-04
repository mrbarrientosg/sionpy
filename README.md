# Sionpy (State Simulation)

Implementacion del Algoritmo Muzero, el agoritmo se siguie del mismo paper, exceptuando la implementacion del PriorityBuffer y Reanalyze. [paper](https://arxiv.org/abs/1911.08265) [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)

# Instalación

Deben instalar las depencias del archivo requirements.txt.

```bash
pip install -r requirements.txt
```

# Ejecutar

Para ejecutar el codigo debe ingresar el siguiente comando:
```bash
python -m sionpy --game cartpole
```

# Configuración

Para agregar otro juego se debe crear una carpeta con el nombre del juego dentro de la carpeta games, y dentro de esta tiene que haber un archivo config.toml y game.py, puede ver los ejemplos de juegos que hay en la carpeta.

# Referencias

* Schrittwieser, Julian et al. (Feb. 21, 2020). “Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model”. [cs, stat]. arXiv:1911.08265