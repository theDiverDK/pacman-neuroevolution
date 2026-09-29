"""
Neuroevolution Pac-Man
======================
50 Pac-Man games run side by side, every Pac-Man driven by its own deep neural
network.  The network (12 -> 24 -> 16 -> 8 -> 1, tanh) is a "direction scorer": it is
applied to each of the four directions (up/down/left/right) and Pac-Man takes the
best-scoring open one.  A Pac-Man that is caught by a ghost is stopped.  There is no
time limit: clearing all pellets starts a fresh level (score is kept) and the game
goes on until the Pac-Man is caught.  When every Pac-Man is dead the best brains are
crossed over and slightly mutated to form the next generation.

Controls
    1 2 3 4 5   speed 1x / 2x / 5x / 20x / MAX
    SPACE       pause
    N           end the current generation now
    S           save the best brain seen so far (best_brain.npy)
    ESC / Q     quit

Run:   python pacman_ai.py            (add --resume to seed from best_brain.npy)
       python pacman_ai.py --headless 30   (train 30 generations, no window)
"""
import argparse
import os
import random
import sys
import time
from collections import deque

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
POP = 50                    # games / Pac-Men simulated at once
STEP_HZ = 10                # game steps per second (1 step = 1 tile for Pac-Man)
STARVE_STEPS = 150          # no pellet eaten for this long -> Pac-Man gives up (0 = off)

# The brain is one small network applied to every direction: its 12 inputs describe a single
# direction (9 numbers) plus 3 global numbers, its output is that direction's score.
N_IN = 39                                # numbers produced by Game.features() per step
LAYERS = [12, 24, 16, 8, 1]              # input, 3 hidden layers, output score
FIT_TIME = 3.0                           # fitness reward per step survived (on top of score)

# Next generation: [0] best brain unchanged, [1] best brain with a little mutation, rest bred
CLONE_MUTATION = (0.03, 0.03)   # (rate, sigma) for the mutated copy of the best brain
PARENT_POOL = 12            # children are bred from this many top brains
# (rate, sigma) of the mutation: children cycle through coarse, medium and fine settings
MUTATIONS = [(0.10, 0.30), (0.05, 0.10), (0.03, 0.03)]

FRIGHT_STEPS = 50           # how long ghosts stay edible after a power pellet
GHOST_SPEED = 0.8           # tiles per step (Pac-Man moves 1.0)
FRIGHT_SPEED = 0.5
MODES = [(True, 50), (False, 150), (True, 50), (False, 150), (True, 40), (False, 10 ** 9)]  # (scatter?, steps)

SAVE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_brain.npy")

# --------------------------------------------------------------------------
# Maze
# --------------------------------------------------------------------------
LAYOUT = [
    "###################",
    "#........#........#",
    "#O##.###.#.###.##O#",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#....#...#...#....#",
    "####.###.#.###.####",
    "   #.#.......#.#   ",
    "####.#.##-##.#.####",
    "      .#GGG#.      ",
    "####.#.#####.#.####",
    "   #.#.......#.#   ",
    "####.###.#.###.####",
    "#........#........#",
    "#.##.###.#.###.##.#",
    "#O.#.....P.....#.O#",
    "##.#.#.#####.#.#.##",
    "#....#...#...#....#",
    "#.######.#.######.#",
    "#.................#",
    "###################",
]
W, H = len(LAYOUT[0]), len(LAYOUT)
assert all(len(r) == W for r in LAYOUT), "maze rows must all have the same width"

PAC_DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]        # action order: up, down, left, right
GHOST_DIRS = [(0, -1), (-1, 0), (0, 1), (1, 0)]      # classic tie-break order
BIG = 1000.0


def _build_maze():
    wall = [[c == "#" for c in row] for row in LAYOUT]
    house, door, start = set(), None, None
    for y, row in enumerate(LAYOUT):
        for x, c in enumerate(row):
            if c == "G":
                house.add((x, y))
            elif c == "-":
                door = (x, y)
            elif c == "P":
                start = (x, y)

    def walkable(x, y):
        return not wall[y][x] and (x, y) not in house and (x, y) != door

    # flood fill from Pac-Man's start; anything unreachable becomes wall
    reach, todo = {start}, deque([start])
    while todo:
        x, y = todo.popleft()
        for dx, dy in PAC_DIRS:
            nx, ny = (x + dx) % W, y + dy
            if 0 <= ny < H and walkable(nx, ny) and (nx, ny) not in reach:
                reach.add((nx, ny))
                todo.append((nx, ny))
    for y in range(H):
        for x in range(W):
            if (x, y) not in reach and (x, y) not in house and (x, y) != door:
                wall[y][x] = True

    cells = sorted(reach, key=lambda p: (p[1], p[0]))
    cid = [[-1] * W for _ in range(H)]
    for i, (x, y) in enumerate(cells):
        cid[y][x] = i
    n = len(cells)

    nbr = np.full((n, 4), -1, dtype=np.int64)
    for i, (x, y) in enumerate(cells):
        for a, (dx, dy) in enumerate(PAC_DIRS):
            nx, ny = (x + dx) % W, y + dy
            if 0 <= ny < H and cid[ny][nx] >= 0:
                nbr[i, a] = cid[ny][nx]

    # all-pairs walking distance (BFS from every cell)
    dist = np.full((n, n), 999.0, dtype=np.float32)
    for s in range(n):
        dist[s, s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if v >= 0 and dist[s, v] == 999.0:
                    dist[s, v] = dist[s, u] + 1
                    q.append(int(v))

    ghost_nbrs = {}
    for (x, y) in cells:
        ghost_nbrs[(x, y)] = []
        for dx, dy in GHOST_DIRS:
            nx, ny = (x + dx) % W, y + dy
            if 0 <= ny < H and cid[ny][nx] >= 0:
                ghost_nbrs[(x, y)].append((nx, ny, dx, dy))

    pen_dot = np.full(n, BIG, dtype=np.float32)
    pen_pow = np.full(n, BIG, dtype=np.float32)
    for i, (x, y) in enumerate(cells):
        if LAYOUT[y][x] == ".":
            pen_dot[i] = 0
        elif LAYOUT[y][x] == "O":
            pen_pow[i] = 0

    return dict(wall=wall, house=house, door=door, start=start, cells=cells, cid=cid, nbr=nbr,
                dist=dist, ghost_nbrs=ghost_nbrs, pen_dot=pen_dot, pen_pow=pen_pow)


_M = _build_maze()
WALL = _M["wall"]
HOUSE = _M["house"]
DOOR = _M["door"]
START = _M["start"]
CELLS = _M["cells"]
CID = _M["cid"]
NBR = _M["nbr"]
DIST = _M["dist"]
GNBR = _M["ghost_nbrs"]
PEN_DOT0 = _M["pen_dot"]
PEN_POW0 = _M["pen_pow"]
N_DOTS = int((PEN_DOT0 == 0).sum())
START_CID = CID[START[1]][START[0]]
HOUSE_Y = next(iter(HOUSE))[1]
DOOR_X, DOOR_Y = DOOR
EXIT_Y = DOOR_Y - 1

# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------
G_HOUSE, G_LEAVING, G_OUT = 0, 1, 2
GHOST_CORNERS = [(W - 2, 1), (1, 1), (W - 2, H - 2), (1, H - 2)]
GHOST_RELEASE = [0, 10, 40, 80]
GHOST_START = [(DOOR_X, EXIT_Y), (DOOR_X, HOUSE_Y), (DOOR_X - 1, HOUSE_Y), (DOOR_X + 1, HOUSE_Y)]



class Ghost:
    __slots__ = ("kind", "x", "y", "px", "py", "dx", "dy", "state", "release", "fright", "acc", "rev")

    def __init__(self, kind):
        self.kind = kind
        self.x, self.y = GHOST_START[kind]
        self.px, self.py = self.x, self.y
        self.dx, self.dy = -1, 0
        self.state = G_OUT if kind == 0 else G_HOUSE
        self.release = GHOST_RELEASE[kind]
        self.fright = False
        self.acc = 0.0
        self.rev = False


class Game:
    """One Pac-Man maze.  Pac-Man makes one tile move per step."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.cell = START_CID
        self.x, self.y = START
        self.dir = (-1, 0)
        self.pen_dot = PEN_DOT0.copy()
        self.pen_pow = PEN_POW0.copy()
        self.dots_left = N_DOTS
        self.score = 0
        self.t = 0
        self.alive = True
        self.cause = None
        self.since_eat = 0
        self.fright_t = 0
        self.combo = 0
        self.level = 1
        self._reset_actors()
        self.pellets_eaten = 0
        self.ghosts_eaten = 0
        self.last_action = -1
        self.eaten = []         # cells cleared since the renderer last looked
        self.refilled = False

    def _reset_actors(self):
        """Pac-Man and the ghosts back to their starting positions (new game or new level)."""
        self.cell = START_CID
        self.x, self.y = START
        self.pprev = START
        self.dir = (-1, 0)
        self.fright_t = 0
        self.combo = 0
        self.since_eat = 0
        self.mode_i = 0
        self.mode_t = MODES[0][1]
        self.scatter = MODES[0][0]
        self.ghosts = [Ghost(k) for k in range(4)]
        self.last_action = -1

    # -- brain inputs ------------------------------------------------------
    def features(self):
        c = self.cell
        nb = NBR[c]
        legal = nb >= 0
        nbc = np.where(legal, nb, c)
        dn = DIST[nbc]                                  # (4, N) distances from each neighbour cell

        danger, edible = [], []
        for g in self.ghosts:
            if g.state == G_HOUSE:
                continue
            gc = CID[g.y][g.x]
            if gc >= 0:
                (edible if g.fright else danger).append(gc)

        f = np.zeros(N_IN, dtype=np.float32)
        f[0:4] = legal

        dp = np.where(legal, (dn + self.pen_dot).min(axis=1), BIG)
        m = dp.min()
        if m < BIG / 2:
            f[4:8] = dp == m                            # step that leads to nearest pellet
        f[8:12] = np.where(legal, 1.0 / (1.0 + dp), 0.0)

        if danger:
            f[12:16] = np.where(legal, 1.0 / (1.0 + dn[:, danger].min(axis=1)), 0.0)
            f[25] = 1.0 / (1.0 + DIST[c, danger].min())
        if edible:
            f[16:20] = np.where(legal, 1.0 / (1.0 + dn[:, edible].min(axis=1)), 0.0)
        f[20:24] = np.where(legal, 1.0 / (1.0 + (dn + self.pen_pow).min(axis=1)), 0.0)

        f[24] = self.fright_t / FRIGHT_STEPS
        f[26] = self.dots_left / N_DOTS

        if self.last_action >= 0:
            f[27 + self.last_action] = 1.0                      # what I did last step
        # room to run: cells whose shortest path from here starts with this step, limited to
        # cells Pac-Man reaches before any dangerous ghost does (and how many pellets are in it)
        reach = (dn + 1.0) == DIST[c][None, :]
        if danger:
            reach &= (dn + 1.0) < DIST[danger].min(axis=0)[None, :]
        reach &= legal[:, None]
        f[31:35] = reach.sum(axis=1) / 60.0
        f[35:39] = (reach & (self.pen_dot == 0)[None, :]).sum(axis=1) / 20.0
        return f

    # -- simulation --------------------------------------------------------
    def step(self, action):
        self.t += 1
        self.last_action = action
        for g in self.ghosts:
            g.px, g.py = g.x, g.y
        self.pprev = (self.x, self.y)

        nb = NBR[self.cell, action]
        if nb >= 0:
            self.cell = int(nb)
            self.x, self.y = CELLS[self.cell]
            self.dir = PAC_DIRS[action]
        self._eat()

        self._collide()
        if not self.alive:
            return

        if self.fright_t > 0:
            self.fright_t -= 1
            if self.fright_t == 0:
                for g in self.ghosts:
                    g.fright = False
        self.mode_t -= 1
        if self.mode_t <= 0 and self.mode_i + 1 < len(MODES):
            self.mode_i += 1
            self.scatter, self.mode_t = MODES[self.mode_i]
            for g in self.ghosts:
                g.rev = True

        for g in self.ghosts:
            g.acc += FRIGHT_SPEED if g.fright else GHOST_SPEED
            if g.acc >= 1.0:
                g.acc -= 1.0
                self._move_ghost(g)
        self._collide()
        if not self.alive:
            return

        if STARVE_STEPS and self.since_eat > STARVE_STEPS:
            self.alive, self.cause = False, "starved"

    def _eat(self):
        c = self.cell
        self.since_eat += 1
        if self.pen_dot[c] == 0:
            self.pen_dot[c] = BIG
            self.dots_left -= 1
            self.pellets_eaten += 1
            self.score += 10
            self.since_eat = 0
            self.eaten.append(c)
        elif self.pen_pow[c] == 0:
            self.pen_pow[c] = BIG
            self.score += 50
            self.since_eat = 0
            self.eaten.append(c)
            self.fright_t = FRIGHT_STEPS
            self.combo = 0
            for g in self.ghosts:
                if g.state != G_HOUSE:
                    g.fright = True
                    g.rev = True
        if self.dots_left == 0:                         # level cleared: fresh maze, score is kept
            self.pen_dot = PEN_DOT0.copy()
            self.pen_pow = PEN_POW0.copy()
            self.dots_left = N_DOTS
            self.score += 500
            self.level += 1
            self.refilled = True
            self.eaten.clear()
            self._reset_actors()

    def _collide(self):
        px, py = self.x, self.y
        for g in self.ghosts:
            if g.state == G_HOUSE:
                continue
            hit = (g.x == px and g.y == py) or \
                  ((g.x, g.y) == self.pprev and (g.px, g.py) == (px, py))
            if not hit:
                continue
            if g.fright:
                self.score += 200 * (2 ** self.combo)
                self.combo += 1
                self.ghosts_eaten += 1
                g.x, g.y = DOOR_X, HOUSE_Y
                g.px, g.py = g.x, g.y
                g.state, g.fright = G_HOUSE, False
                g.release = self.t + 20
            else:
                self.alive, self.cause = False, "caught"
                return

    def _move_ghost(self, g):
        if g.state == G_HOUSE:
            if self.t >= g.release:
                g.state = G_LEAVING
            return
        if g.state == G_LEAVING:
            if g.y == HOUSE_Y and g.x != DOOR_X:
                g.x += 1 if g.x < DOOR_X else -1
            else:
                g.y -= 1
                if g.y <= EXIT_Y:
                    g.y = EXIT_Y
                    g.state = G_OUT
                    g.dx, g.dy = 0, -1
            return

        allopts = GNBR[(g.x, g.y)]
        if g.rev:
            g.rev = False
            back = [o for o in allopts if o[2] == -g.dx and o[3] == -g.dy]
            if back:
                self._go(g, back[0])
                return
        opts = [o for o in allopts if not (o[2] == -g.dx and o[3] == -g.dy)] or allopts
        rng = self.rng
        if g.fright or (g.kind == 2 and rng.random() < 0.35):
            self._go(g, rng.choice(opts))
            return
        tx, ty = self._target(g)
        best, bd = opts[0], 1e9
        for o in opts:
            d = (o[0] - tx) ** 2 + (o[1] - ty) ** 2
            if d < bd:
                best, bd = o, d
        self._go(g, best)

    @staticmethod
    def _go(g, o):
        g.x, g.y, g.dx, g.dy = o

    def _target(self, g):
        if self.scatter:
            return GHOST_CORNERS[g.kind]
        if g.kind == 1:
            return self.x + 4 * self.dir[0], self.y + 4 * self.dir[1]
        if g.kind == 3 and (g.x - self.x) ** 2 + (g.y - self.y) ** 2 < 64:
            return GHOST_CORNERS[3]
        return self.x, self.y

    @property
    def fitness(self):
        return self.score + FIT_TIME * self.t


# --------------------------------------------------------------------------
# Neural networks & evolution
# --------------------------------------------------------------------------
def _layout_genes():
    """Offsets of every layer inside the flat genome, plus a neuron id for each gene
    (so crossover can swap whole neurons instead of scattering single weights)."""
    layers, neuron_of, off, nid = [], [], 0, 0
    for i in range(len(LAYERS) - 1):
        n_in, n_out = LAYERS[i], LAYERS[i + 1]
        layers.append((off, off + n_in * n_out, off + n_in * n_out + n_out, n_in, n_out))
        neuron_of.append(np.tile(np.arange(nid, nid + n_out), n_in))    # weights, row-major (in, out)
        neuron_of.append(np.arange(nid, nid + n_out))                   # biases
        off += n_in * n_out + n_out
        nid += n_out
    return layers, np.concatenate(neuron_of), off, nid


LAYER_SLICES, NEURON_OF_GENE, N_GENES, N_NEURONS = _layout_genes()


def random_population(rng, n=POP):
    pop = np.zeros((n, N_GENES), dtype=np.float32)
    for w0, w1, b1, n_in, n_out in LAYER_SLICES:
        pop[:, w0:w1] = rng.normal(0, 1.0 / np.sqrt(n_in) * 1.5, (n, n_in * n_out))
    return pop


def unpack(pop):
    """Batched weight views: [(W (P,in,out), b (P,out)), ...]"""
    out = []
    for w0, w1, b1, n_in, n_out in LAYER_SLICES:
        out.append((pop[:, w0:w1].reshape(len(pop), n_in, n_out), pop[:, w1:b1]))
    return out


def evolve(pop, fitness, rng):
    order = np.argsort(-fitness)
    ranked = pop[order]
    new = np.empty_like(pop)
    new[0] = ranked[0]                                  # the best, untouched
    rate, sigma = CLONE_MUTATION
    new[1] = ranked[0] + (rng.random(N_GENES) < rate) * rng.normal(0, sigma, N_GENES)   # best + a little mutation
    weights = np.arange(PARENT_POOL, 0, -1, dtype=np.float64)
    weights /= weights.sum()
    for i in range(2, len(pop)):
        a, b = rng.choice(PARENT_POOL, 2, replace=False, p=weights)
        take_b = rng.random(N_NEURONS) < 0.5
        child = np.where(take_b[NEURON_OF_GENE], ranked[b], ranked[a])
        rate, sigma = MUTATIONS[i % len(MUTATIONS)]
        mutate = rng.random(N_GENES) < rate
        child = child + mutate * rng.normal(0, sigma, N_GENES)
        new[i] = child
    return new.astype(np.float32)


class Generation:
    def __init__(self, pop, seed):
        self.pop = pop
        self.seed = seed
        self.games = [Game(seed) for _ in range(len(pop))]    # same ghost luck for everybody
        self.layers = unpack(pop)
        self.t = 0

    @property
    def alive_count(self):
        return sum(g.alive for g in self.games)

    @property
    def done(self):
        return self.alive_count == 0

    def step(self):
        idx = [i for i, g in enumerate(self.games) if g.alive]
        if not idx:
            return
        ia = np.array(idx)
        x = np.stack([self.games[i].features() for i in idx])
        # the same network scores each of the 4 directions: (A,4,12) = 9 per-direction + 3 global inputs
        d = np.arange(4)
        per_dir = np.stack([x[:, 0 + d], x[:, 4 + d], x[:, 8 + d], x[:, 12 + d], x[:, 16 + d],
                            x[:, 20 + d], x[:, 27 + d], x[:, 31 + d], x[:, 35 + d]], axis=2)
        glob = np.repeat(x[:, None, [24, 25, 26]], 4, axis=1)
        h = np.concatenate([per_dir, glob], axis=2)
        last = len(self.layers) - 1
        for li, (w, b) in enumerate(self.layers):
            h = np.matmul(h, w[ia]) + b[ia][:, None, :]
            if li < last:
                h = np.tanh(h)
        scores = h[:, :, 0]
        out = np.where(x[:, 0:4] > 0, scores, -1e9)            # never pick a wall
        for i, a in zip(idx, out.argmax(axis=1)):
            self.games[i].step(int(a))
        self.t += 1

    def fitness(self):
        return np.array([g.fitness for g in self.games])


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
class Renderer:
    COLS, ROWS, GAP, HUD = 10, 5, 6, 74
    GHOST_COLORS = [(255, 40, 40), (255, 184, 255), (0, 235, 235), (255, 184, 82)]

    def __init__(self, pg, cell=None):
        self.pg = pg
        info = pg.display.Info()
        if cell is None:
            cw = (info.current_w - 40 - self.GAP * (self.COLS + 1)) / (self.COLS * W)
            ch = (info.current_h - 110 - self.HUD - self.GAP * (self.ROWS + 1)) / (self.ROWS * H)
            cell = max(4, min(10, int(min(cw, ch))))
        self.cell = cell
        self.bw, self.bh = W * cell, H * cell
        self.width = self.COLS * (self.bw + self.GAP) + self.GAP
        self.height = self.HUD + self.ROWS * (self.bh + self.GAP) + self.GAP
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption("Neuroevolution Pac-Man")
        self.font = pg.font.SysFont("consolas,dejavusansmono,couriernew", 15)
        self.small = pg.font.SysFont("consolas,dejavusansmono,couriernew", max(9, cell + 2))
        self.base = self._build_base()
        self.views = []
        self.text_cache = {}

    def _build_base(self):
        pg, c = self.pg, self.cell
        s = pg.Surface((self.bw, self.bh))
        s.fill((0, 0, 0))
        fill, edge = (14, 14, 80), (70, 90, 255)
        for y in range(H):
            for x in range(W):
                if not WALL[y][x]:
                    continue
                s.fill(fill, (x * c, y * c, c, c))
        for y in range(H):
            for x in range(W):
                if not WALL[y][x]:
                    continue
                for dx, dy in PAC_DIRS:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < W and 0 <= ny < H and not WALL[ny][nx]:
                        px, py = x * c, y * c
                        if dx == 1:
                            pg.draw.line(s, edge, (px + c - 1, py), (px + c - 1, py + c - 1))
                        elif dx == -1:
                            pg.draw.line(s, edge, (px, py), (px, py + c - 1))
                        elif dy == 1:
                            pg.draw.line(s, edge, (px, py + c - 1), (px + c - 1, py + c - 1))
                        else:
                            pg.draw.line(s, edge, (px, py), (px + c - 1, py))
        pg.draw.line(s, (255, 150, 200), (DOOR_X * c, DOOR_Y * c + c // 2),
                     (DOOR_X * c + c - 1, DOOR_Y * c + c // 2), 2)
        return s

    def new_generation(self, gen):
        pg, c = self.pg, self.cell
        self.views = []
        for _ in gen.games:
            surf = self.base.copy()
            self.views.append({"surf": surf, "dead": False})
        for g, v in zip(gen.games, self.views):
            self._draw_pellets(v["surf"], g)

    def _draw_pellets(self, surf, g):
        pg, c = self.pg, self.cell
        for i, (x, y) in enumerate(CELLS):
            if g.pen_dot[i] == 0:
                pg.draw.circle(surf, (255, 220, 190), (x * c + c // 2, y * c + c // 2), max(1, c // 6))
            elif g.pen_pow[i] == 0:
                pg.draw.circle(surf, (255, 255, 255), (x * c + c // 2, y * c + c // 2), max(2, int(c * 0.38)))

    def _text(self, key, txt, font, color):
        k = (key, txt, color)
        s = self.text_cache.get(k)
        if s is None:
            if len(self.text_cache) > 2000:
                self.text_cache.clear()
            s = self.text_cache[k] = font.render(txt, True, color)
        return s

    def draw(self, gen, info):
        pg, c = self.pg, self.cell
        scr = self.screen
        scr.fill((8, 8, 14))

        leader, lead_score = -1, -1
        for i, g in enumerate(gen.games):
            if g.alive and g.score > lead_score:
                leader, lead_score = i, g.score

        for i, (g, v) in enumerate(zip(gen.games, self.views)):
            surf = v["surf"]
            if g.refilled:
                g.refilled = False
                surf.blit(self.base, (0, 0))
                self._draw_pellets(surf, g)
            if g.eaten:
                for cid in g.eaten:
                    x, y = CELLS[cid]
                    surf.fill((0, 0, 0), (x * c, y * c, c, c))
                g.eaten.clear()
            if not g.alive and not v["dead"]:
                v["dead"] = True
                surf.fill((85, 85, 85), special_flags=pg.BLEND_RGB_MULT)
                col = (255, 60, 60) if g.cause == "caught" else (255, 170, 40)
                x0, y0 = g.x * c, g.y * c
                pg.draw.line(surf, col, (x0, y0), (x0 + c, y0 + c), 2)
                pg.draw.line(surf, col, (x0 + c, y0), (x0, y0 + c), 2)

            ox = self.GAP + (i % self.COLS) * (self.bw + self.GAP)
            oy = self.HUD + self.GAP + (i // self.COLS) * (self.bh + self.GAP)
            scr.blit(surf, (ox, oy))

            if g.alive:
                self._draw_actors(g, ox, oy)
            scr.blit(self._text("s", str(g.score) if g.level == 1 else f"L{g.level} {g.score}", self.small, (255, 255, 255) if g.alive else (150, 150, 150)),
                     (ox + 2, oy))
            if i == leader:
                pg.draw.rect(scr, (255, 215, 0), (ox - 2, oy - 2, self.bw + 4, self.bh + 4), 2)

        self._draw_hud(gen, info)

    def _draw_actors(self, g, ox, oy):
        pg, c = self.pg, self.cell
        r = c / 2 - 0.5
        for gh in g.ghosts:
            cx, cy = ox + gh.x * c + c / 2, oy + gh.y * c + c / 2
            if gh.fright:
                col = (245, 245, 245) if (g.fright_t <= 15 and g.t % 2) else (40, 60, 255)
            else:
                col = self.GHOST_COLORS[gh.kind]
            pg.draw.rect(self.screen, col, (cx - r, cy - r, 2 * r, 2 * r + 0.5),
                         border_top_left_radius=int(r), border_top_right_radius=int(r))
        cx, cy = ox + g.x * c + c / 2, oy + g.y * c + c / 2
        pg.draw.circle(self.screen, (255, 235, 0), (cx, cy), r)
        if g.t % 2 == 0:
            dx, dy = g.dir
            px, py = -dy, dx
            pg.draw.polygon(self.screen, (0, 0, 0), [
                (cx, cy),
                (cx + dx * (r + 1) + px * r * 0.9, cy + dy * (r + 1) + py * r * 0.9),
                (cx + dx * (r + 1) - px * r * 0.9, cy + dy * (r + 1) - py * r * 0.9)])

    def _draw_hud(self, gen, info):
        pg, scr = self.pg, self.screen
        alive = gen.alive_count
        fits = gen.fitness()
        secs = gen.t / STEP_HZ
        top = max(g.level for g in gen.games)
        l1 = (f"Generation {info['gen']:<4d}  time {secs:5.1f}s   alive {alive:2d}/{len(gen.games)}   top level {top}   "
              f"speed {info['speed_label']}{'  [PAUSED]' if info['paused'] else ''}")
        l2 = (f"best now {fits.max():7.1f}   avg now {fits.mean():7.1f}   "
              f"last gen best/avg {info['last_best']:7.1f}/{info['last_avg']:7.1f}   all-time best {info['all_best']:7.1f}")
        l3 = "1-5 speed | SPACE pause | N next generation | S save best brain | ESC quit"
        for i, (t, col) in enumerate([(l1, (255, 235, 0)), (l2, (220, 220, 220)), (l3, (120, 130, 150))]):
            scr.blit(self._text("h%d" % i, t, self.font, col), (10, 6 + i * 20))

        hist = info["history"]
        gx, gy, gw, gh_ = self.width - 330, 8, 316, 56
        pg.draw.rect(scr, (20, 22, 34), (gx, gy, gw, gh_))
        if len(hist) >= 2:
            top = max(max(b for b, _ in hist), 1.0)
            n = len(hist)
            for k, col in ((0, (255, 235, 0)), (1, (0, 200, 230))):
                pts = [(gx + 4 + (gw - 8) * i / (n - 1), gy + gh_ - 4 - (gh_ - 8) * h[k] / top)
                       for i, h in enumerate(hist)]
                pg.draw.lines(scr, col, False, pts, 2)
        scr.blit(self._text("gl", "best", self.small, (255, 235, 0)), (gx + 4, gy + 1))
        scr.blit(self._text("ga", "avg", self.small, (0, 200, 230)), (gx + 44, gy + 1))


# --------------------------------------------------------------------------
# Main loops
# --------------------------------------------------------------------------
def save_brain(genome):
    np.save(SAVE_FILE, genome)


def initial_population(rng, resume):
    pop = random_population(rng)
    if resume and os.path.exists(SAVE_FILE):
        best = np.load(SAVE_FILE).astype(np.float32)
        pop[0] = best
        for i in range(1, len(pop)):
            pop[i] = best + (rng.random(N_GENES) < 0.05) * rng.normal(0, 0.1, N_GENES)
        print("resumed from", SAVE_FILE)
    return pop


def run_headless(gens, resume):
    rng = np.random.default_rng()
    pop = initial_population(rng, resume)
    all_best = -1.0
    for n in range(1, gens + 1):
        t0 = time.perf_counter()
        gen = Generation(pop, random.randrange(1 << 30))
        while not gen.done:
            gen.step()
        fit = gen.fitness()
        b = int(fit.argmax())
        if fit[b] > all_best:
            all_best = float(fit[b])
            save_brain(pop[b])
        best = gen.games[b]
        print(f"gen {n:3d}  best {fit.max():7.1f}  avg {fit.mean():6.1f}  "
              f"pellets {best.pellets_eaten:3d} ghosts {best.ghosts_eaten}  steps {gen.t:3d}  "
              f"{time.perf_counter() - t0:4.1f}s")
        pop = evolve(pop, fit, rng)


def run_gui(resume):
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    import pygame as pg
    pg.init()
    ren = Renderer(pg)
    clock = pg.time.Clock()

    rng = np.random.default_rng()
    pop = initial_population(rng, resume)

    speeds = {pg.K_1: 1, pg.K_2: 2, pg.K_3: 5, pg.K_4: 20, pg.K_5: None}
    speed = 1
    paused = False
    step_dt = 1.0 / STEP_HZ
    accum = 0.0

    info = dict(gen=1, last_best=0.0, last_avg=0.0, all_best=0.0, history=[], speed_label="1x",
                paused=False)
    all_best_genome = None

    def start(pop):
        g = Generation(pop, random.randrange(1 << 30))
        ren.new_generation(g)
        return g

    gen = start(pop)
    force_end = False
    running = True
    while running:
        frame_dt = clock.tick(60) / 1000.0
        for e in pg.event.get():
            if e.type == pg.QUIT:
                running = False
            elif e.type == pg.KEYDOWN:
                if e.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif e.key in speeds:
                    speed = speeds[e.key]
                elif e.key == pg.K_SPACE:
                    paused = not paused
                elif e.key == pg.K_n:
                    force_end = True
                elif e.key == pg.K_s and all_best_genome is not None:
                    save_brain(all_best_genome)

        info["speed_label"] = "MAX" if speed is None else f"{speed}x"
        info["paused"] = paused

        if not paused:
            t0 = time.perf_counter()
            if speed is None:
                budget = 0.035
                n_steps = 10 ** 9
            else:
                accum += min(frame_dt, 0.1) * speed
                n_steps = int(accum / step_dt)
                accum -= n_steps * step_dt
                budget = 0.045
            while n_steps > 0 or force_end:
                if gen.done or force_end:
                    fit = gen.fitness()
                    b = int(fit.argmax())
                    if fit[b] > info["all_best"]:
                        info["all_best"] = float(fit[b])
                        all_best_genome = gen.pop[b].copy()
                        save_brain(all_best_genome)
                    info["last_best"], info["last_avg"] = float(fit.max()), float(fit.mean())
                    info["history"].append((info["last_best"], info["last_avg"]))
                    pop = evolve(gen.pop, fit, rng)
                    info["gen"] += 1
                    gen = start(pop)
                    force_end = False
                    break
                gen.step()
                n_steps -= 1
                if time.perf_counter() - t0 > budget:
                    break

        ren.draw(gen, info)
        pg.display.flip()
    pg.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Neuroevolution Pac-Man")
    ap.add_argument("--resume", action="store_true", help="seed the population from best_brain.npy")
    ap.add_argument("--headless", type=int, metavar="GENS", help="train without a window for GENS generations")
    args = ap.parse_args()
    if args.headless:
        run_headless(args.headless, args.resume)
    else:
        run_gui(args.resume)
