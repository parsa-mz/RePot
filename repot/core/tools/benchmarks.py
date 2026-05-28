from __future__ import annotations

from repot.core.env import Env
from repot.envs.blocksworld import BlocksWorldEnv
from repot.envs.blocksworld_pddl import BlocksWorldPDDLEnv
from repot.envs.checker_jumping import CheckerJumpingEnv
from repot.envs.hanoi import TowerOfHanoiEnv
from repot.envs.river_crossing import RiverCrossingEnv


def make_env(name: str) -> Env:
    """Construct the `Env` subclass for `name`, accepting a few common aliases."""
    aliases = {
        "hanoi": "tower_of_hanoi",
        "tower_of_hanoi": "tower_of_hanoi",
        "checker": "checker_jumping",
        "checker_jumping": "checker_jumping",
        "river": "river_crossing",
        "river_crossing": "river_crossing",
        "blocks": "blocksworld",
        "blocksworld": "blocksworld",
        "blocksworld_pddl": "blocksworld_pddl",
        "planbench_blocksworld": "blocksworld_pddl",
    }
    key = aliases.get(name, name)
    if key == "tower_of_hanoi":
        return TowerOfHanoiEnv()
    if key == "checker_jumping":
        return CheckerJumpingEnv()
    if key == "river_crossing":
        return RiverCrossingEnv()
    if key == "blocksworld":
        return BlocksWorldEnv()
    if key == "blocksworld_pddl":
        return BlocksWorldPDDLEnv()
    raise ValueError(f"Unknown environment: {name}")
