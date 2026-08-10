import sys
import time
from pathlib import Path
import gurobipy as gp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bls.config import (
    INSTANCE_PATH,
    TIME_LIMIT,
    THREADS,
    OUTPUT_FLAG,
    SEED,
    NUMERIC_FOCUS,
    MIP_GAP,
    USE_BLS,
    VERBOSE,
    MIN_LP_POINTS
)
from bls.solver import GuroBI_BLS


def main():
    global_start_time = time.time()

    instance_path = INSTANCE_PATH
    if not instance_path.exists():
        if not instance_path.parent.exists():
            raise FileNotFoundError(
                f"Instance directory not found: '{instance_path.parent}'. "
                f"Please ensure the instances folder exists."
            )
        raise FileNotFoundError(
            f"Instance file '{instance_path.name}' not found in directory '{instance_path.parent}'."
        )

    mip = gp.read(str(instance_path))

    mip.Params.OutputFlag = OUTPUT_FLAG
    mip.Params.Threads = THREADS
    mip.Params.Seed = SEED
    mip.Params.NumericFocus = NUMERIC_FOCUS
    mip.Params.MIPGap = MIP_GAP

    mip.Params.Heuristics = 0.0
    mip.Params.Presolve = 0
    mip.Params.Cuts = 0

    solver_backend = GuroBI_BLS(
        mip,
        global_start_time=global_start_time,
        time_limit=TIME_LIMIT,
        use_bls=USE_BLS,
        verbose=VERBOSE,
        min_lp_points=MIN_LP_POINTS
    )
    solver_backend.solve()
    run_stats = solver_backend.statistics()

    print("\n" + "=" * 60)
    print("Final Optimization Results Summary")
    print("=" * 60)
    for key, value in run_stats.items():
        if isinstance(value, float):
            print(f"{key:22s}: {value:.6f}")
        else:
            print(f"{key:22s}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
