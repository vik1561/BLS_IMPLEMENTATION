import math
import itertools
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .config import EPS, BLS_MAX_CANDIDATES, BLS_MAX_EXTENSION, VERBOSE
from .utils import round_integer_part
from .line_search import LineSearch


class BLSController:
    """
    Branch-Line-Search heuristic controller engine.
    Manages base points, line evaluation combinations, lattice scanning, and feasibility intervals.
    """
    def __init__(self, model, integer_indices, bnb=None, verbose=VERBOSE):
        self.model = model
        self.is_minimize = (self.model.ModelSense == GRB.MINIMIZE)
        self.integer_indices = list(integer_indices)
        self.bnb = bnb
        self.verbose = verbose
        self.global_base_points = []
        
        # DEDUPLICATION TRACKING STORES
        self.evaluated_combinations = set()    # Stores pairs of unique base point IDs
        self.evaluated_lattice_points = set()   # Stores unique point hashes (coordinates as string/tuple)
        
        self.best_point = None
        self.best_objective = None
        
        self.line = LineSearch()
        self.max_candidates = BLS_MAX_CANDIDATES
        self.max_extension = BLS_MAX_EXTENSION
        self._cache_model_data()

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _cache_model_data(self):
        self.vars = self.model.getVars()
        self.lb = np.array([-np.inf if v.LB <= -GRB.INFINITY else v.LB for v in self.vars], dtype=float)
        self.ub = np.array([np.inf if v.UB >= GRB.INFINITY else v.UB for v in self.vars], dtype=float)
        self.A = self.model.getA()
        self.constrs = self.model.getConstrs()
        self.rhs = np.array([c.RHS for c in self.constrs], dtype=float)
        self.sense = [c.Sense for c in self.constrs]
        obj = self.model.getObjective()
        coeffs = np.zeros(self.model.NumVars, dtype=float)
        for term in range(obj.size()):
            var = obj.getVar(term)
            coeffs[var.index] += obj.getCoeff(term)
        self.c = coeffs
        self.obj_const = obj.getConstant()
        self.base_activity = None
        self.base_slope = None
        self.obj_ra = None
        self.obj_d = None

    def _get_point_hash(self, point):
        return tuple(np.round(point, 6))

    def process_siblings(self, lp_pool, cb_model=None):
        if self.bnb and self.bnb.is_time_limit_exceeded():
            return []

        all_feasible_points = []
        newly_added_count = 0

        self._log("\n   ==========================================================")
        self._log("   [ROUNDED POINTS EVALUATION STAGE]")
        self._log("   ==========================================================")

        for i, lp in enumerate(lp_pool):
            self._log(f"\n   -> LP Solution {i+1}: {np.round(lp, 4)}")

            c = round_integer_part(lp, self.integer_indices, 'ceil', self.lb, self.ub)
            f = round_integer_part(lp, self.integer_indices, 'floor', self.lb, self.ub)
            r = round_integer_part(lp, self.integer_indices, 'round', self.lb, self.ub)

            current_candidates = [('ceil', c), ('floor', f), ('round', r)]

            for mode_name, p in current_candidates:
                # Check whether the rounded point is already present in the global registry
                if not any(np.allclose(p, gp_pt, atol=EPS) for gp_pt in self.global_base_points):
                    # Add to global registry
                    self.global_base_points.append(p)
                    newly_added_count += 1

                    # Mark as already evaluated in the lattice point hash structure
                    point_hash = self._get_point_hash(p)
                    self.evaluated_lattice_points.add(point_hash)

                    # Evaluate the rounded point itself
                    is_feas = self.is_feasible(p)
                    if is_feas:
                        obj = self.objective_value(point=p)
                        self._log(f"      [{mode_name.upper()} POINT] Point: {np.round(p, 4)} | Feasible: True | Objective: {obj:.6f}")
                        all_feasible_points.append((None, p, obj))
                        self.update_best(p, obj)

                        # If better than incumbent, submit immediately to Gurobi
                        if self.bnb and self.bnb._is_better(obj):
                            target_model = cb_model or self.model
                            self._log(f"      [INCUMBENT IMPROVEMENT] Rounded point objective ({obj:.6f}) improves current incumbent ({self.bnb.best_incumbent:.6f}). Submitting immediately...")
                            self.bnb.submit_solution(target_model, p)
                    else:
                        self._log(f"      [{mode_name.upper()} POINT] Point: {np.round(p, 4)} | Feasible: False")

        self._log(f"\n   [UNIQUE GLOBAL POINT REGISTRY] Accumulated elements: {len(self.global_base_points)} (+{newly_added_count} new)")
        for idx, bp in enumerate(self.global_base_points):
            self._log(f"      Registry Point #{idx}: {np.round(bp, 4)}")
        self._log("   ==========================================================\n")

        combination_counter = 1
        for idx_A, idx_B in itertools.combinations(range(len(self.global_base_points)), 2):
            if self.bnb and self.bnb.is_time_limit_exceeded():
                self._log("   [TIME LIMIT REACHED] Aborting further line evaluations.")
                break

            comb_key = tuple(sorted((idx_A, idx_B)))
            if comb_key in self.evaluated_combinations:
                combination_counter += 1
                continue
                
            RA = self.global_base_points[idx_A]
            RB = self.global_base_points[idx_B]
            
            if np.allclose(RA, RB, atol=EPS):
                continue
                
            self.line.set_base_points(RA, RB)
            self.base_activity = np.asarray(self.A @ RA)
            self.base_slope = np.asarray(self.A @ self.line.direction)
            self.obj_ra = float(self.c @ RA + self.obj_const)
            self.obj_d = float(self.c @ self.line.direction)

            line_feasible = self.search_line()
            all_feasible_points.extend(line_feasible)
            
            self.evaluated_combinations.add(comb_key)
            combination_counter += 1

        return all_feasible_points

    def search_line(self):
        if self.bnb and self.bnb.is_time_limit_exceeded():
            return []

        if self.should_discard_parallel_infeasible():
            self._log("      [LINE SEARCH] Discarded line due to parallel infeasibility check.")
            return []

        interval = self.feasible_t_interval()
        if interval is None:
            self._log("      [LINE SEARCH] Continuous feasible t-interval is empty.")
            return []

        t_low, t_high = interval
        t_values = self.lattice_t_values(t_low, t_high)
        if not t_values:
            self._log(f"      [LINE RESULT] No lattice points in feasible interval [{t_low:.6f}, {t_high:.6f}]")
            return []
        feasible_points = []

        self._log(f"      [LINE SCAN] Feasible t-interval [{t_low:.4f}, {t_high:.4f}]. Scanning lattice points:")
        for t in t_values:
            if self.bnb and self.bnb.is_time_limit_exceeded():
                self._log("      [TIME LIMIT REACHED] Terminating current line search scan.")
                break

            point = self.line.point(t)
            point_hash = self._get_point_hash(point)
            
            if point_hash in self.evaluated_lattice_points:
                continue
            
            is_feas = self.is_feasible(point, t)
            objective = self.objective_value(t)
            
            self._log(f"         Lattice t={t:+.4f} -> Point: {np.round(point, 4)} | Feasible: {is_feas} | Objective: {objective:.6f}")
            
            self.evaluated_lattice_points.add(point_hash)
            if not is_feas:
                continue
                
            feasible_points.append((t, point, objective))
            self.update_best(point, objective, t)

        return feasible_points

    def feasible_t_interval(self):
        RA = self.line.RA
        D = self.line.direction
        low, high = -math.inf, math.inf

        def add_upper(value):
            nonlocal high
            high = min(high, value)

        def add_lower(value):
            nonlocal low
            low = max(low, value)

        def restrict_linear(alpha, beta, sense):
            if abs(beta) <= EPS:
                if sense == "<" and alpha > EPS: return False
                if sense == ">" and alpha < -EPS: return False
                if sense == "=" and abs(alpha) > EPS: return False
                return True
            
            root = -alpha / beta   
            if sense == "<":
                if beta > 0: add_upper(root)
                else: add_lower(root)
            elif sense == ">":
                if beta > 0: add_lower(root)
                else: add_upper(root)
            else:
                add_lower(root)
                add_upper(root)
            return low <= high + EPS
        
        # check model constraints
        for activity, slope, rhs, sense in zip(self.base_activity, self.base_slope, self.rhs, self.sense):
            if not restrict_linear(activity - rhs, slope, sense):
                return None

        # check variable bounds
        for x0, dx, lb, ub in zip(RA, D, self.lb, self.ub):
            if np.isfinite(lb) and not restrict_linear(x0 - lb, dx, ">"): return None
            if np.isfinite(ub) and not restrict_linear(x0 - ub, dx, "<"): return None

        if low > high + EPS:
            return None
        return low, high

    def lattice_t_values(self, low, high):
        if not np.isfinite(low): low = -self.max_extension
        if not np.isfinite(high): high = self.max_extension

        gcd_step = 0
        for idx in self.integer_indices:
            delta = int(round(self.line.direction[idx]))
            gcd_step = math.gcd(gcd_step, abs(delta))
        if gcd_step == 0: gcd_step = 1

        lo_k = math.ceil(low * gcd_step - EPS)
        hi_k = math.floor(high * gcd_step + EPS)
        if lo_k > hi_k: return []

        keys = list(range(lo_k, hi_k + 1))
        if len(keys) > self.max_candidates:
            keep = set()
            edge = max(1, self.max_candidates // 4)
            keep.update(range(lo_k, min(hi_k, lo_k + edge) + 1))
            keep.update(range(max(lo_k, hi_k - edge), hi_k + 1))
            for k in np.linspace(lo_k, hi_k, self.max_candidates):
                keep.add(int(round(k)))
            keys = sorted(keep, key=lambda k: (abs(k), k))[:self.max_candidates]

        return [k / gcd_step for k in keys]

    def should_discard_parallel_infeasible(self):
        if self.base_activity is None or self.base_slope is None:
            self.base_activity = np.asarray(self.A @ self.line.RA)
            self.base_slope = np.asarray(self.A @ self.line.direction)
        act_a = self.base_activity
        act_b = self.base_activity + self.base_slope
        for a, b, rhs, sense in zip(act_a, act_b, self.rhs, self.sense):
            slack_a, slack_b = a - rhs, b - rhs
            if sense == "<":
                if slack_a > EPS and slack_b > EPS and abs(slack_a - slack_b) <= EPS: return True
            elif sense == ">":
                if slack_a < -EPS and slack_b < -EPS and abs(slack_a - slack_b) <= EPS: return True
            else:
                if abs(slack_a) > EPS and abs(slack_b) > EPS and abs(abs(slack_a) - abs(slack_b)) <= EPS: return True
        return False

    def is_feasible(self, point, t=None):
        if np.any(point < self.lb - EPS) or np.any(point > self.ub + EPS): return False
        if t is not None and self.base_activity is not None and self.base_slope is not None:
            activity = self.base_activity + t * self.base_slope
        else:
            activity = np.asarray(self.A @ point)
        for lhs, rhs, sense in zip(activity, self.rhs, self.sense):
            if sense == "<" and lhs > rhs + EPS: return False
            if sense == ">" and lhs < rhs - EPS: return False
            if sense == "=" and abs(lhs - rhs) > EPS: return False
        return True

    def objective_value(self, t=None, point=None):
        if t is not None and self.obj_ra is not None and self.obj_d is not None:
            return self.obj_ra + self.obj_d * t
        if point is not None:
            return float(self.c @ point + self.obj_const)
        raise ValueError("Either t or point must be provided to objective_value")

    def update_best(self, point, objective, t=None):
        if self.best_objective is None:
            self.best_objective = objective
            self.best_point = point
        elif self.is_minimize:
            if objective < self.best_objective:
                self.best_objective = objective
                self.best_point = point
        else:
            if objective > self.best_objective:
                self.best_objective = objective
                self.best_point = point

