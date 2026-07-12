#!/usr/bin/env python3
"""Generate SLURM job-array params files + a manifest for the training suite.

ASP-Failure Thesis campaign (thesis_plan.md §2.1).
Writes one params file per (phase, family, memory-tier) so that every array
submission has uniform --mem.

Params line formats:
  single-agent : SCRIPT SEED NUM_ENVS MAX_ITERS SAVE_INTERVAL EXP_NAME [EXTRA]
  self-play    : SCRIPT SEED NUM_ENVS MAX_ITERS SAVE_INTERVAL EXP_NAME [EXTRA]
  gym          : SCRIPT SEED NUM_ENVS PUSH_NSTEPS MAX_ITERS SAVE_INTERVAL EXP_NAME [EXTRA]
  validate     : VALIDATOR EXP_NAME [EXTRA]

Manifest line: PARAMS_FILE  TEMPLATE  MEM  TIME  THROTTLE  PHASE
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_DIR = os.path.join(HERE, "params")

# ── Budget matching: 528 envs x 3000 iters x 15 pushes = 23.76M pushes ────────
BUDGET_PUSHES = 528 * 3000 * 15
def iters_for(envs):
    return round(BUDGET_PUSHES / (envs * 15))

MEM_TIER = {256: "12G", 512: "12G", 528: "12G", 1024: "12G", 2048: "48G"}
GYM_MEM = {64: "32G", 256: "32G"}
TIME = "23:00:00"

TPL_SINGLE = "hpc/arrays/train_single_agent.slurm"
TPL_SELF = "hpc/arrays/train_self_play.slurm"
TPL_GYM = "hpc/arrays/train_gym.slurm"
TPL_VAL = "hpc/arrays/validate.slurm"
TPL_GYM_VAL = "hpc/arrays/validate_gym.slurm"

# ── Canonical ASP = ABC-on (β=0.5 from yaml). NO --no_abc here.  The ablation
#    arms add --no_abc where needed (phase5_ablate, phase6_abcpc_off). ──────────
SELFPLAY_COMMON = ("--alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2 ")
SELFPLAY_EXTRA = (SELFPLAY_COMMON + "--char_length 0.07 --dpose_threshold 0.055")
G_EXTRA = SELFPLAY_EXTRA + " --alice_reward_scale 0.5"
DISC_EXTRA = (SELFPLAY_COMMON + "--char_length 0.0 --dpose_threshold 0.05")
H_EXTRA = DISC_EXTRA + " --alice_reward_scale 0.5"
I_EXTRA = SELFPLAY_EXTRA + " --alice_reward_scale 0.5"
BASE_EXTRA = "--rel-obs --rel-act"
SAVE = 100

# model -> (script, family, extra, validator, val_extra)
MODELS = {
    "pbrsA": ("train_a_pbrs_simple.py", "single", "", "validate_push.py", BASE_EXTRA),
    "pbrsB": ("train_b_pbrs_curriculum.py", "single", "", "validate_push.py", BASE_EXTRA),
    "base":  ("train_push.py", "single", BASE_EXTRA, "validate_push.py", BASE_EXTRA),
    "pbrsE": ("train_e_pbrs_asp_dpose.py", "self", SELFPLAY_EXTRA, "validate_push_asp.py",
              "--dpose-obs --char-length 0.07 --dpose-threshold 0.055"),
    "pbrsG": ("train_g_tasp_dpose.py", "self", G_EXTRA, "validate_push_asp.py",
              "--dpose-obs --char-length 0.07 --dpose-threshold 0.055"),
    # discF (Model F, position-only disc). Canonical = ABC-on (no --no_abc).
    "discF": ("train_f_pbrs_asp_disc.py", "self", DISC_EXTRA, "validate_push_asp.py",
              "--dpose-obs --char-length 0.0 --dpose-threshold 0.05 --scene-set disc"),
    # taspH — TASP disc (time-based Alice, disc).
    "taspH": ("train_h_tasp_disc.py", "self", H_EXTRA, "validate_push_asp.py",
              "--dpose-obs --char-length 0.0 --dpose-threshold 0.05 --scene-set disc"),
    # taspI — TASP T-block + Bob time penalty (full symmetric Sukhbaatar reward).
    "taspI": ("train_i_tasp_dpose_bobpen.py", "self", I_EXTRA, "validate_push_asp.py",
              "--dpose-obs --char-length 0.07 --dpose-threshold 0.055"),
}


class Suite:
    def __init__(self):
        self.buckets = {}   # (phase, family, mem) -> list[train line]
        self.val = {}       # phase -> list[validate line]   (Isaac)
        self.gymval = {}    # phase -> list[validate line]   (gym)

    def add(self, phase, model, seed, envs, extra_override=None, name_suffix=""):
        script, family, extra, validator, val_extra = MODELS[model]
        iters = iters_for(envs)
        exp = f"{model}_e{envs}_i{iters}_s{seed}{name_suffix}"
        ex = extra if extra_override is None else extra_override
        line = f"{script} {seed} {envs} {iters} {SAVE} {exp} {ex}".rstrip()
        mem = MEM_TIER[envs]
        self.buckets.setdefault((phase, family, mem), []).append(line)
        self.val.setdefault(phase, []).append(f"{validator} {exp} {val_extra}".rstrip())

    def add_gym(self, phase, model_script, seed, envs, push_nsteps, exp):
        iters = 3000
        line = f"{model_script} {seed} {envs} {push_nsteps} {iters} {SAVE} {exp}"
        mem = GYM_MEM.get(envs, "32G")
        self.buckets.setdefault((phase, "gym", mem), []).append(line)
        self.gymval.setdefault(phase, []).append(f"validate_pusht_gym.py {exp}")

    def add_xeval(self, phase, exp, val_extra):
        self.val.setdefault(phase, []).append(
            f"validate_push_asp.py {exp} {val_extra}".rstrip())

    def write(self):
        os.makedirs(PARAMS_DIR, exist_ok=True)
        import glob
        for old in glob.glob(os.path.join(PARAMS_DIR, "*.txt")):
            os.remove(old)
        tpl = {"single": TPL_SINGLE, "self": TPL_SELF, "gym": TPL_GYM}
        manifest = []
        for (phase, family, mem), lines in sorted(self.buckets.items()):
            fname = f"{phase}_{family}_{mem}.txt"
            with open(os.path.join(PARAMS_DIR, fname), "w") as f:
                f.write("\n".join(lines) + "\n")
            throttle = 4
            manifest.append(f"hpc/params/{fname} {tpl[family]} {mem} {TIME} {throttle} {phase}")
        for phase, lines in sorted(self.val.items()):
            fname = f"{phase}_validate.txt"
            with open(os.path.join(PARAMS_DIR, fname), "w") as f:
                f.write("\n".join(lines) + "\n")
            manifest.append(f"hpc/params/{fname} {TPL_VAL} 12G 06:00:00 4 {phase}_validate")
        for phase, lines in sorted(self.gymval.items()):
            fname = f"{phase}_validate.txt"
            with open(os.path.join(PARAMS_DIR, fname), "w") as f:
                f.write("\n".join(lines) + "\n")
            manifest.append(f"hpc/params/{fname} {TPL_GYM_VAL} 32G 04:00:00 4 {phase}_validate")
        with open(os.path.join(PARAMS_DIR, "manifest.txt"), "w") as f:
            f.write("# PARAMS_FILE  TEMPLATE  MEM  TIME  THROTTLE  PHASE\n")
            f.write("\n".join(manifest) + "\n")
        print(f"Wrote params to {PARAMS_DIR}")
        for (phase, family, mem), lines in sorted(self.buckets.items()):
            print(f"  {phase:20s} {family:7s} {mem:4s} : {len(lines):3d} jobs")


def build():
    s = Suite()
    seeds3 = [7, 42, 123]
    seeds5 = [7, 42, 123, 202, 707]
    envs4 = [256, 512, 1024, 2048]

    # ── Phase 1 (anchor+efficiency) -------------------------------------------
    # Single-agent PPO-Baseline and PBRS across {256,512,1024,2048} × 3 seeds.
    # Earns the "task solvable; 4×-fewer-environments" result.
    for model in ("pbrsA", "base"):
        for e in envs4:
            for seed in seeds3:
                s.add("phase1_anchor", model, seed, e)

    # ── Phase 2 (seed CIs) ─────────────────────────────────────────────────────
    # Headline success rate table with confidence intervals.
    # Baseline, PBRS(A), Curriculum(B), E, G at 528 × 5 seeds.
    for model in ("pbrsA", "pbrsB", "pbrsE", "pbrsG", "base"):
        for seed in seeds5:
            s.add("phase2_ci", model, seed, 528)

    # ── Phase 3 (coupled-gate isolation) ───────────────────────────────────────
    # disc F and disc-TASP H at 528 × 5 seeds — the key "ASP succeeds when gate
    # is position-only" result. ABC-on canonical for both.
    for model in ("discF", "taspH"):
        for seed in seeds5:
            s.add("phase3_disc", model, seed, 528)

    # ── Phase 4 (ASP scale sweep — endpoints only) ─────────────────────────────
    # E and G across {256, 2048} × 3 seeds. "Does scale rescue ASP?"
    for model in ("pbrsE", "pbrsG"):
        for e in (256, 2048):
            for seed in seeds3:
                s.add("phase4_scale", model, seed, e)

    # ── Phase 5 (component ablations) ──────────────────────────────────────────
    # E and G: no-ABC, no-GE — which component matters?
    # no-ABC arm: drop ABC, keep GE.
    for model in ("pbrsE", "pbrsG"):
        ablate_abc = SELFPLAY_COMMON + "--char_length 0.07 --dpose_threshold 0.055 --no_abc"
        for seed in seeds3:
            s.add("phase5_ablate", model, seed, 528, extra_override=ablate_abc, name_suffix="_noabc")
    # no-GE arm: keep ABC on (canonical), drop GoalEncoder.
    for model in ("pbrsE", "pbrsG"):
        ablate_noge = SELFPLAY_COMMON + "--char_length 0.07 --dpose_threshold 0.055 --no_ge"
        for seed in seeds3:
            s.add("phase5_ablate", model, seed, 528, extra_override=ablate_noge, name_suffix="_noge")

    # ── Phase 6 (ABC positive control) ─────────────────────────────────────────
    # disc F: ABC-on (3 seeds) AND ABC-off (3 seeds) — "ABC helps when easy, not
    # when coupled." ABC-on reuses discF's canonical extra (DISC_EXTRA, no --no_abc).
    for seed in seeds3:
        s.add("phase6_abcpc", "discF", seed, 528, name_suffix="_abc")
    disc_abc_off = SELFPLAY_COMMON + "--char_length 0.0 --dpose_threshold 0.05 --no_abc"
    for seed in seeds3:
        s.add("phase6_abcpc", "discF", seed, 528, extra_override=disc_abc_off, name_suffix="_noabc")

    # ── Phase 7 (reward-structure — Bob-penalty) ───────────────────────────────
    # Bob-penalty I at 528 × 5 seeds. Full symmetric Sukhbaatar reward.
    for seed in seeds5:
        s.add("phase7_bobpen", "taspI", seed, 528)

    # ── Phase 8 (cross-environment gym A/B/C) ──────────────────────────────────
    # A/B are AsyncVectorEnv (spawn N workers) — 256 envs OOMs at 32G; cap at 64.
    # C is single-process ASP, no spawn workers — 64/256 both safe.
    for name, script in (("gymA", "train_a_gym_pbrs_simple.py"),
                         ("gymB", "train_b_gym_pbrs_curriculum.py")):
        for envs in (64,):
            for pns in (15, 45):
                s.add_gym("phase8_gym", script, 42, envs, pns, f"{name}_e{envs}_p{pns}_s42")
    for envs in (64, 256):
        s.add_gym("phase8_gym", "train_c_gym_pbrs_asp.py", 42, envs, 15, f"gymC_e{envs}_p15_s42")

    # ── Cross-eval (E, G on disc scenes; validation-only) ─────────────────────
    xeval_extra = "--dpose-obs --char-length 0.0 --dpose-threshold 0.05 --scene-set disc"
    for model in ("pbrsE", "pbrsG"):
        for seed in seeds3:
            s.add_xeval("phase_xeval", f"{model}_e528_i3000_s{seed}", xeval_extra)

    s.write()


if __name__ == "__main__":
    build()
