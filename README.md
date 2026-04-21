# nextPlayer

A JAX-based object-centric reinforcement learning system combining:
- **3D POV Environment**: First-person view room with bouncing balls
- **modSTOVE**: Modified STOVE model with 128D object representations (3D physics + 119D latent)
- **STOVEplayer**: SwiGLU-based LLM wrapper for agent control

## Model Sizes

| Component | Parameters | Description |
|-----------|------------|-------------|
| **modSTOVE** | ~1.5M | Image encoder + slot attention + GNN dynamics |
| **STOVEplayer** | ~175M | SwiGLU transformer (8+4 layers, 768D, 32K vocab) |

Breakdown:
- modSTOVE: CNN encoder (~300K) + Slot encoder (~800K) + Dynamics GNN (~400K)
- STOVEplayer: Embeddings (~50M) + Encoder 8 layers (~76M) + Decoder 4 layers (~48M) + Output (~25M)

## Architecture

```
Environment (JAX) → 128x128 POV Image → modSTOVE → 7 slots × 128D → STOVEplayer → Actions
```

### modSTOVE Object Representation (128D per slot)
| Component | Dims | Description |
|-----------|------|-------------|
| Position  | 3    | (x, y, z) in 3D space |
| Velocity  | 3    | (vx, vy, vz) |
| Size      | 3    | (sx, sy, sz) scale factors |
| Latent    | 119  | Appearance + unstructured features |

### STOVEplayer
- SwiGLU-based transformer encoder (8 layers) and decoder (4 layers)
- Slot context via additive projection (physics + content embeddings)
- 768D embedding dimension, 6 attention heads, 32K vocabulary

## Environment

A first-person 3D room where:
- **Walls**: Light blue, occasional green-tinted mirrors
- **Agent**: Beige cylinder (only visible in mirror reflections)
- **Balls**: Indigo, slow-moving, elastic collisions
- **Goal**: Catch balls (collision = +1 score, ball disappears)

### Actions
- `FORWARD` / `STOP`: Movement control
- `TURN_LEFT` / `TURN_RIGHT` / `STOP_TURNING`: Rotation control

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Play the environment manually
```bash
python -m nextPlayer.gui.play
```

Controls:
- `W/↑` Forward, `S/↓` Stop
- `A/←` Turn left, `D/→` Turn right, `SPACE` Stop turning
- `R` Reset, `P` Pause, `ESC` Quit

### Watch autonomous simulation
```bash
python -m nextPlayer.gui.play --watch
```

### Pretrain modSTOVE
```bash
python -m nextPlayer.modSTOVE_pretraining
```

## Project Structure

```
nextPlayer/
├── environment/          # JAX 3D environment
│   ├── room.py          # Room geometry + rendering
│   ├── physics.py       # Ball physics (elastic collisions)
│   ├── agent_controller.py  # Agent movement
│   └── env.py           # Gymnax-style wrapper
├── agent/
│   ├── modSTOVE/        # Object-centric world model
│   │   ├── image_model.py
│   │   ├── slot_encoder.py
│   │   ├── dynamics.py
│   │   └── model.py
│   └── stove_player.py  # SwiGLU LLM wrapper
├── gui/                 # Interactive debugging
│   ├── viewer.py
│   └── play.py
└── modSTOVE_pretraining.py
```

## References

modSTOVE is based on the STOVE architecture:

```bibtex
@inproceedings{kossen2020structured,
  title={Structured Object-Aware Physics Prediction for Video Modeling and Planning},
  author={Kossen, Jannik and Stelzner, Karl and Hussing, Marcel and Voelcker, Claas and Kersting, Kristian},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=B1e-kxSKDH}
}
```

Key modifications from original STOVE:
- Extended to 3D (position, velocity, size each 3D instead of 2D)
- Larger latent space (119D vs 12D) for richer appearance modeling
- 128x128 input resolution (vs 32x32)
- 7 object slots (vs 3)
