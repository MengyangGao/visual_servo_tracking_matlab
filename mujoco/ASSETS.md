# MuJoCo Asset Sources

This project currently uses:

- Google DeepMind MuJoCo Menagerie as a git submodule for the Franka Emika Panda robot model: `mujoco/vendor/mujoco_menagerie/franka_emika_panda/panda.xml`.
- Built-in MJCF primitive and compound targets defined in `src/mujoco_servo/targets.py`.

Useful open-source MJCF asset sources checked for future expansion:

- Google DeepMind MuJoCo Menagerie: curated high-quality MuJoCo robot models with XML and mesh assets.
- robosuite: MuJoCo-based robot learning framework with MJCF object model abstractions and examples such as XML object classes.

The current target library intentionally stays primitive/compound-only so the visual-servo demo remains lightweight, deterministic, and easy to test. Larger object asset libraries can be added later behind the same `TargetSpec` interface.
