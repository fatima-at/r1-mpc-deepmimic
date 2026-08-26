# Source of this model

`R1_C++.xml`, `scene.xml` and `meshes/` are the Unitree R1 description, taken
unmodified from:

    https://github.com/unitreerobotics/unitree_mujoco
    path: unitree_robots/r1
    commit: sparse checkout of branch main

Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics").
Licensed BSD 3-Clause; see `LICENSE` in this directory.

The R1 is not present in `mujoco_menagerie`.

No file in this directory is modified by this project. The changes this project
needs (torque actuators to position servos, foot sensors, implicit integrator)
are applied in memory at load time via `MjSpec` in `mpc/env.py`.
