import math
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB, GurobiError

from .config import TIME_LIMIT, EPS, USE_BLS, MIN_LP_POINTS, VERBOSE
from .controller import BLSController


class GuroBI_BLS:
    """
    Gurobi Solver Wrapper integrating Branch-Line-Search heuristic via MIP Callbacks.
    Can run either with BLS heuristic enabled (use_bls=True) or as pure Gurobi solver (use_bls=False).
    """
    def __init__(
        self,
        model,
        global_start_time=None,
        time_limit=TIME_LIMIT,
        use_bls=USE_BLS,
        min_lp_points=MIN_LP_POINTS,
        verbose=VERBOSE
    ):
        self.original_model = model
        self.use_bls = use_bls
        self.verbose = verbose
        
        self.global_start_time = global_start_time if global_start_time is not None else time.time()
        self.time_limit = time_limit
        
        self.objective_sense = self.original_model.ModelSense
        self.is_minimize = (self.objective_sense == GRB.MINIMIZE)
        self.tolerance = EPS
        
        self.bls_calls = 0
        self.bls_solutions_found = 0
        self.bls_improvements = 0
        self.best_incumbent = math.inf if self.is_minimize else -math.inf
        
        self.integer_indices = [
            i for i, var in enumerate(self.original_model.getVars()) if var.VType != GRB.CONTINUOUS  
        ]
        
        self.bls = BLSController(self.original_model, self.integer_indices, self, verbose=self.verbose)
        
        self.submitted_solutions = set()
        self.lp_solution_pool = []
        self.min_lp_points = min_lp_points

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _get_point_hash(self, point):
        return tuple(np.round(point, 6))

    def elapsed_time(self):
        return time.time() - self.global_start_time

    def is_time_limit_exceeded(self):
        return self.elapsed_time() >= self.time_limit

    def _is_better(self, obj_new, obj_old=None):
        if obj_old is None: obj_old = self.best_incumbent
        if self.is_minimize: return obj_new < obj_old - self.tolerance
        return obj_new > obj_old + self.tolerance
    
    def update_incumbent(self, objective, source):
        if self._is_better(objective):
            if source == "cbUseSolution":
                self.bls_improvements += 1
                self._log(f"      [BLS ACCEPTED IMPROVEMENT] Objective improved to: {objective:.6f}")
            else:
                self._log(f"      [INCUMBENT SYNC] Objective updated to: {objective:.6f} ({source})")

            self.best_incumbent = objective

    def _mip_callback(self, model, where):
        try:
            if self.is_time_limit_exceeded():
                if not getattr(self, '_time_limit_logged', False):
                    self._log("\n[GLOBAL TIME LIMIT] Exceeded in callback. Terminating Gurobi solver...")
                    self._time_limit_logged = True
                model.terminate()
                return

            if where == GRB.Callback.MIPNODE:
                node_count = int(model.cbGet(GRB.Callback.MIPNODE_NODCNT))
                self._log(f"\n[CALLBACK EVENT] MIPNODE Triggered (Global Gurobi Node Count: {node_count})")
                
                status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
                if status == GRB.OPTIMAL:
                    lp_sol = model.cbGetNodeRel(model.getVars())
                    lp_solution = np.array(lp_sol, dtype=float)

                    # Store only unique LP relaxations
                    if not any(np.allclose(lp_solution, p, atol=EPS) for p in self.lp_solution_pool):
                        self.lp_solution_pool.append(lp_solution.copy())

                    self._log(f"      [LP POOL] {len(self.lp_solution_pool)} LP solutions collected.")

                    # Trigger BLS search when minimum LP solution pool is reached
                    if len(self.lp_solution_pool) >= self.min_lp_points:
                        self._apply_bls_on_siblings(model, None, None)
                
            elif where == GRB.Callback.MIPSOL:
                mipsol_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                self._log(f"\n[CALLBACK EVENT] MIPSOL Found! Incumbent objective level synchronized to: {mipsol_obj:.6f}")
                self.update_incumbent(mipsol_obj, source="MIPSOL")
                
        except Exception as e:
            print(f"Error executing callback pipeline: {e}")
    
    def submit_solution(self, model, candidate):
        """
        Submits candidate solution to Gurobi using cbSetSolution and cbUseSolution.
        Logs acceptance/rejection and updates incumbent stats if accepted.
        Skips submission if candidate has already been submitted previously.
        """
        cand_hash = self._get_point_hash(candidate)
        if cand_hash in self.submitted_solutions:
            self._log(f"\n   [SUBMISSION SKIPPED] Candidate point: {np.round(candidate, 4)} was already submitted previously.")
            return None

        self.submitted_solutions.add(cand_hash)
        self._log(f"\n   [SUBMISSION PREPARATION] Candidate point: {np.round(candidate, 4)}")
        try:
            vars_ = model.getVars()
            values = [float(v) for v in candidate]
            model.cbSetSolution(vars_, values)

            obj_value = model.cbUseSolution()

            if obj_value == GRB.INFINITY or obj_value == -GRB.INFINITY:
                self._log("   [SUBMISSION RESULT] -> REJECTED by Gurobi (Cutoff constraint violation or structurally infeasible).")
            else:
                self._log(f"   [SUBMISSION RESULT] -> ACCEPTED by Gurobi! Evaluated Objective: {obj_value:.6f}")
                self.update_incumbent(obj_value, source="cbUseSolution")
            return obj_value
        except GurobiError as e:
            self._log(f"   [SUBMISSION RESULT] -> REJECTED due to direct engine error: {e}")
            return None

    def _apply_bls_on_siblings(self, model, lp_sol_1, lp_sol_2):
        try:
            if self.is_time_limit_exceeded():
                model.terminate()
                return

            self.bls_calls += 1
            feasible_points = self.bls.process_siblings(self.lp_solution_pool, cb_model=model)

            if feasible_points:
                self.bls_solutions_found += len(feasible_points)
                best_point = min(feasible_points, key=lambda x: x[2] if self.is_minimize else -x[2])
                candidate = best_point[1]
                
                self._log(f"\n   [SUBMISSION PREPARATION] Best valid point across ALL global evaluations: {np.round(candidate, 4)} with objective: {best_point[2]:.6f}")
                self.submit_solution(model, candidate)

        except Exception as e:
            print(f"Error in BLS sister processing phase: {e}")

            
    def solve(self):
        rem_time = max(0.0, self.time_limit - self.elapsed_time())
        if rem_time <= 0:
            print("[GLOBAL TIME LIMIT] Reached before optimization could start.")
            return self.original_model

        self.original_model.Params.TimeLimit = rem_time
        if self.use_bls:
            self._log("\n[OPTIMIZATION RUN] Running Gurobi with Branch-Line-Search (BLS) Callbacks...")
            self.original_model.optimize(self._mip_callback)
        else:
            self._log("\n[OPTIMIZATION RUN] Running Pure Gurobi (BLS Disabled)...")
            self.original_model.optimize()
    
    def statistics(self):
        return {
            "use_bls": self.use_bls,
            "status": self.original_model.Status,
            "best_objective": self.original_model.ObjVal if self.original_model.SolCount else (self.best_incumbent if np.isfinite(self.best_incumbent) else None),
            "best_bound": self.original_model.ObjBound if self.original_model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.NODE_LIMIT] or self.original_model.SolCount else None,
            "mip_gap": self.original_model.MIPGap if self.original_model.SolCount else None,
            "total_time_sec": self.elapsed_time(),
            "iterations": self.original_model.IterCount,
            "nodes_processed": self.original_model.NodeCount,
            "bls_calls": self.bls_calls,
            "bls_solutions_found": self.bls_solutions_found,
            "bls_improvements": self.bls_improvements,
            "best_bls_objective": self.bls.best_objective,
            "best_bls_point": self.bls.best_point.tolist() if self.bls.best_point is not None else None,
        }
