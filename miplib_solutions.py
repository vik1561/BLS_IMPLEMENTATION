import os
import json
import re
import urllib.request
from pathlib import Path

# Paths
BLS_DIR = Path(__file__).resolve().parent
DATA_DIR = BLS_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DEFAULT_SOLU_PATH = DATA_DIR / "miplib2017.solu"
DEFAULT_CACHE_PATH = DATA_DIR / "solutions_cache.json"


def normalize_instance_name(name):
    """Strips directory path, extensions (.mps, .mps.gz, .lp), and duplicate tags like ' (1)'."""
    base = os.path.basename(str(name))
    base = re.sub(r"\.(mps|gz|lp)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\.(mps|gz|lp)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"\s*\(\d+\)$", "", base)  # Remove ' (1)'
    return base.strip().lower()


def load_solutions_cache(cache_path=DEFAULT_CACHE_PATH):
    """Loads persistent solution cache JSON file."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_solution_to_cache(instance_name, obj_val, status="OPTIMAL", source="Local Cache", cache_path=DEFAULT_CACHE_PATH):
    """Saves a solution objective value to persistent local JSON cache for future runs."""
    norm_name = normalize_instance_name(instance_name)
    cache = load_solutions_cache(cache_path)
    cache[norm_name] = {
        "raw_name": os.path.basename(str(instance_name)),
        "obj_val": float(obj_val) if obj_val is not None else None,
        "status": status,
        "source": source,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as ex:
        print(f"[CACHE WARNING] Could not update solution cache: {ex}")


def parse_solu_file(solu_path=DEFAULT_SOLU_PATH):
    """Parses .solu file format (=opt= name val / =best= name val). Returns dict of normalized_name -> (status, obj_val)."""
    solu_db = {}
    if not solu_path.exists():
        return solu_db

    with open(solu_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("Title"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                tag = parts[0].strip()  # =opt=, =best=, =inf=, =unb=, =unkn=
                raw_name = parts[1].strip()
                norm_name = normalize_instance_name(raw_name)
                val = float(parts[2]) if len(parts) >= 3 else None
                solu_db[norm_name] = (tag, val, raw_name)
    return solu_db


def fetch_solution_online(instance_name):
    """Attempts to fetch instance optimal/best objective value from MIPLIB online site."""
    norm_name = normalize_instance_name(instance_name)
    url = f"https://miplib.zib.de/instance/{norm_name}.html"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode("utf-8", errors="ignore")
            match = re.search(r"(?:Objective|Objective value|Optimal value|Value)\s*[:=]?\s*<[^>]+>\s*(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", html, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                return val, "=opt="
    except Exception:
        pass
    return None, None


def get_miplib_solution(instance_name, solu_path=DEFAULT_SOLU_PATH, cache_path=DEFAULT_CACHE_PATH):
    """
    Looks up reference objective value for an instance name.
    Order of priority:
      1st: Check official MIPLIB dataset (.solu file / MIPLIB web)
      2nd: Check persistent JSON cache (solutions_cache.json)
      3rd: Return None so caller can solve using baseline solver if unavailable
    Returns (obj_val, status_str, source_str) or (None, "NOT_FOUND", "N/A")
    """
    norm_name = normalize_instance_name(instance_name)

    # 1st Priority: Check local .solu file (miplib2017.solu)
    solu_db = parse_solu_file(solu_path)
    if norm_name in solu_db:
        tag, val, raw_name = solu_db[norm_name]
        if val is not None:
            status_desc = "OPTIMAL" if tag == "=opt=" else f"BEST_KNOWN_{tag}"
            save_solution_to_cache(instance_name, val, status=status_desc, source="MIPLIB2017 solu", cache_path=cache_path)
            return val, status_desc, "MIPLIB 2017 Solu Dataset"

    # 1st Priority (Fallback): Check online MIPLIB web lookup
    val_online, tag_online = fetch_solution_online(instance_name)
    if val_online is not None:
        save_solution_to_cache(instance_name, val_online, status="OPTIMAL", source="MIPLIB Web Fetch", cache_path=cache_path)
        return val_online, "OPTIMAL", "MIPLIB Online Web"

    # 2nd Priority: Check persistent JSON cache (e.g. past Gurobi baseline solves)
    cache = load_solutions_cache(cache_path)
    if norm_name in cache:
        entry = cache[norm_name]
        return entry.get("obj_val"), entry.get("status", "OPTIMAL"), f"Cache ({entry.get('source', 'Persistent JSON')})"

    # 3rd Priority: Not found (caller will run solver if unavailable)
    return None, "NOT_FOUND", "N/A"
