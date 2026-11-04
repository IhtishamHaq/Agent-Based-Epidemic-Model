# abm.py
import random
from typing import Dict, Tuple, List, Optional, Iterable

import numpy as np
from packages.abm.agent import Agent

Coord = Tuple[int, int]


def von_neumann(x: int, y: int) -> List[Coord]:
    return [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]


class ABM:
    STATUS_TO_CODE = {"S": 0, "I": 1, "R": 2}
    CODE_TO_STATUS = {0: "S", 1: "I", 2: "R"}

    def __init__(
        self,
        num_infected: int,
        beta: float = 0.5,
        gamma: int = 14,
        init_tau: int = 0,
        init_taus: Optional[Dict[Coord, int]] = None,
        position_infected: Optional[List[Coord]] = None,
        min_infected: int = 3, 
    ):
        self.beta = float(beta)
        self.gamma = int(gamma)
        self.min_infected = int(min_infected)
        self.agents: Dict[Coord, Agent] = {}

        if position_infected is not None:
            positions = list(position_infected)
        else:
            width, height = 1, 1
            all_cells = [(x, y) for x in range(width) for y in range(height)]
            if num_infected > len(all_cells):
                raise ValueError(
                    f"Cannot place {num_infected} infected in {width}x{height} grid ({len(all_cells)} cells)."
                )
            positions = random.sample(all_cells, num_infected)

        norm_positions: List[Coord] = []
        seen = set()
        for x, y in positions:
            pos = (int(x), int(y))
            if pos not in seen:
                norm_positions.append(pos)
                seen.add(pos)

        def tau_for(pos: Coord) -> int:
            if init_taus and pos in init_taus:
                return int(init_taus[pos])
            return int(init_tau)

        for pos in norm_positions:
            self.agents[pos] = Agent(
                status=Agent.I,
                beta=self.beta,
                gamma=self.gamma,
                infected_age=tau_for(pos),
            )

        self.counts: List[Dict[str, int]] = []
        self.new_infections_daily: List[int] = []

    @classmethod
    def from_state(
        cls,
        *,
        beta: float,
        gamma: int,
        X: Iterable[int],
        Y: Iterable[int],
        status_codes: Iterable[int],
        infected_ages: Optional[Iterable[int]] = None,
        min_infected: int = 3,  # ✅ NEW
    ) -> "ABM":
        obj: "ABM" = object.__new__(cls)
        obj.beta = float(beta)
        obj.gamma = int(gamma)
        obj.min_infected = int(min_infected)
        obj.agents = {}

        X = np.asarray(X, dtype=int).ravel()
        Y = np.asarray(Y, dtype=int).ravel()
        S = np.asarray(status_codes, dtype=int).ravel()
        if infected_ages is None:
            ages = np.zeros_like(S, dtype=int)
        else:
            ages = np.asarray(infected_ages, dtype=int).ravel()

        if not (len(X) == len(Y) == len(S) == len(ages)):
            raise ValueError("X, Y, status_codes, infected_ages must have the same length.")

        for x, y, sc, age in zip(X, Y, S, ages):
            status = cls.CODE_TO_STATUS.get(int(sc))
            if status is None:
                raise ValueError(
                    f"Unknown status code {sc}. Expected one of {list(cls.CODE_TO_STATUS.keys())}."
                )
            obj.agents[(int(x), int(y))] = Agent(
                status=status,
                beta=obj.beta,
                gamma=obj.gamma,
                infected_age=int(age) if status == Agent.I else 0,
            )

        obj.counts = []
        obj.new_infections_daily = []
        return obj

    def to_arrays(self):
        n = len(self.agents)
        X = np.empty(n, dtype=int)
        Y = np.empty(n, dtype=int)
        S = np.empty(n, dtype=int)
        A = np.empty(n, dtype=int)

        for i, ((x, y), a) in enumerate(self.agents.items()):
            X[i] = int(x)
            Y[i] = int(y)
            S[i] = self.STATUS_TO_CODE.get(a.status, 0)
            A[i] = int(a.days_infected if a.status == Agent.I else 0)

        return X, Y, S, A

    def _state_counts(self):
        s = sum(1 for a in self.agents.values() if a.status == Agent.S)
        i = sum(1 for a in self.agents.values() if a.status == Agent.I)
        r = sum(1 for a in self.agents.values() if a.status == Agent.R)
        return {"S": s, "I": i, "R": r}

    def _ensure_min_infected(self) -> int:
        if self.min_infected <= 0:
            return 0

        infected_now = [pos for pos, a in self.agents.items() if a.status == Agent.I]
        need = self.min_infected - len(infected_now)
        if need <= 0:
            return 0

        sus_positions = [pos for pos, a in self.agents.items() if a.status == Agent.S]
        if not sus_positions:
            return 0

        chosen = random.sample(sus_positions, min(need, len(sus_positions)))
        for pos in chosen:
            a = self.agents[pos]
            a.status = Agent.I
            a.days_infected = 0

        return len(chosen)

    def step(self) -> int:
        coords_to_check = set(self.agents.keys())
        for (x, y), a in self.agents.items():
            if a.status == Agent.I:
                coords_to_check.update(von_neumann(x, y))

        new_agents: Dict[Coord, Agent] = {}
        new_infections = 0

        for (x, y) in coords_to_check:
            current = self.agents.get((x, y))
            if current is None:
                current = Agent(status=Agent.S, beta=self.beta, gamma=self.gamma)

            k_infected = 0
            for nx, ny in von_neumann(x, y):
                nb = self.agents.get((nx, ny))
                if nb is not None and nb.status == Agent.I:
                    k_infected += 1

            prev_status = current.status

            replica = Agent(
                status=current.status,
                beta=current.beta,
                gamma=current.gamma,
                infected_age=current.days_infected,
            )
            replica.step(k_infected)

            if prev_status == Agent.S and replica.status == Agent.I:
                new_infections += 1

            new_agents[(x, y)] = replica

        self.agents = new_agents

        
        added = self._ensure_min_infected()
        new_infections += int(added)

        self.counts.append(self._state_counts())
        self.new_infections_daily.append(int(new_infections))
        return int(new_infections)

    def run_simulation(self, num_steps: int):
        self.counts.append(self._state_counts())
        for _ in range(num_steps):
            self.step()
        return self.counts
