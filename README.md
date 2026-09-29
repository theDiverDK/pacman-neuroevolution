# Neuroevolution Pac-Man

Pac-Man in pygame where 50 games run side by side, each played by its own small neural
network. Pac-Men that get caught by a ghost stop; when all are dead, the best brains are
copied, crossed over and mutated into the next generation. Clearing a level starts a new
one and the score keeps counting.

```
pip install pygame numpy
python pacman_ai.py                 # watch it train
python pacman_ai.py --headless 100  # train without a window
python pacman_ai.py --resume        # start from the saved best_brain.npy
```

Controls: `1`-`5` speed, `SPACE` pause, `N` next generation, `S` save best brain, `ESC` quit.

## The brain
One shared MLP (12 -> 24 -> 16 -> 8 -> 1, tanh, 857 weights) scores each of the four
directions; the best open one is taken. Inputs per direction: open, on the path to the nearest
pellet, closeness of pellet / dangerous ghost / edible ghost / power pellet, last move, room to
run before a ghost arrives, pellets in that room; plus frightened timer, danger at the current
tile and fraction of dots left.

## Evolution
Each generation: the best brain is kept unchanged, one copy of it gets a little mutation, and
the other 48 are bred by neuron-wise crossover of the top 12 with coarse/medium/fine mutation.
Fitness = score + 3 per step survived.
