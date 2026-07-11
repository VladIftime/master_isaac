#!/usr/bin/env python3
"""Generate SLURM job-array params files + a manifest for the training suite.

Writes one params file per (phase, family, memory-tier) so that every array
submission has uniform --mem (a job array shares resource directives across all
its tasks).  Also writes a matching validation params file per phase and a
manifest that submit_all.sh consumes.

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

MEM_TIER = {256: "24G", 512: "24G", 528: "24G", 1024: "32G", 2048: "48G"}
TIME = "23:00:00"

TPL_SINGLE = "hpc/arrays/train_single_agent.slurm"
TPL_SELF = "hpc/arrays/train_self_play.slurm"
TPL_GYM = "hpc/arrays/train_gym.slurm"
TPL_VAL = "hpc/arrays/validate.slurm"

SELFPLAY_EXTRA = ("--alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2 "
                  "--char_length 0.07 --dpose_threshold 0.055 --no_abc")
G_EXTRA = SELFPLAY_EXTRA + " --alice_reward_scale 0.5"
BASE_EXTRA = "--rel-obs --rel-act"          # train_push baseline
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
    # Disc (position-only d_pose) — ABC-on positive control substrate.
    "discF": ("train_f_pbrs_asp_disc.py", "self",
              "--alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2 "
              "--char_length 0.0 --dpose_threshold 0.05 --no_abc",
              "validate_push_asp.py", "--dpose-obs --char-length 0.0 --dpose-threshold 0.05"),
}


class Suite:
    def __init__(self):
        self.buckets = {}   # (phase, family, mem) -> list[train line]
        self.val = {}       # phase -> list[validate line]

    def add(self, phase, model, seed, envs, extra_override=None, name_suffix=""):
        script, family, extra, validator, val_extra = MODELS[model]
        iters = iters_for(envs)
        exp = f"{model}_e{envs}_i{iters}_s{seed}{name_suffix}"
        ex = extra if extra_override is None else extra_override
        if family == "single":
            line = f"{script} {seed} {envs} {iters} {SAVE} {exp} {ex}".rstrip()
        else:
            line = f"{script} {seed} {envs} {iters} {SAVE} {exp} {ex}".rstrip()
        mem = MEM_TIER[envs]
        self.buckets.setdefault((phase, family, mem), []).append(line)
        self.val.setdefault(phase, []).append(f"{validator} {exp} {val_extra}".rstrip())

    def add_gym(self, phase, model_script, seed, envs, push_nsteps, exp):
        iters = 3000
        line = f"{model_script} {seed} {envs} {push_nsteps} {iters} {SAVE} {exp}"
        self.buckets.setdefault((phase, "gym", "16G"), []).append(line)

    def write(self):
        os.makedirs(PARAMS_DIR, exist_ok=True)
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
            manifest.append(f"hpc/params/{fname} {TPL_VAL} 24G 06:00:00 4 {phase}_validate")
        with open(os.path.join(PARAMS_DIR, "manifest.txt"), "w") as f:
            f.write("# PARAMS_FILE  TEMPLATE  MEM  TIME  THROTTLE  PHASE\n")
            f.write("\n".join(manifest) + "\n")
        # report
        print(f"Wrote params to {PARAMS_DIR}")
        for (phase, family, mem), lines in sorted(self.buckets.items()):
            print(f"  {phase:16s} {family:7s} {mem:4s} : {len(lines):3d} jobs")


def build():
    s = Suite()
    seeds3 = [7, 42, 123]
    seeds5 = [7, 42, 123, 202, 707]
    envs4 = [256, 512, 1024, 2048]

    # Phase 1 (E9): self-play scale sweep, E + G, seed 42
    for model in ("pbrsE", "pbrsG"):
        for e in envs4:
            s.add("phase1", model, 42, e)

    # Phase 2 (E2): single-agent env frontier, pbrsA + baseline, 3 seeds
    for model in ("pbrsA", "base"):
        for e in envs4:
            for seed in seeds3:
                s.add("phase2", model, seed, e)

    # Phase 3 (E1): multi-seed CIs @528, all five models, 5 seeds
    for model in ("pbrsA", "pbrsB", "pbrsE", "pbrsG", "base"):
        for seed in seeds5:
            s.add("phase3", model, seed, 528)

    # Phase 4 (E4): self-play ablations from E @528, 3 seeds
    _, _, e_extra, _, _ = MODELS["pbrsE"]
    for tag, flag in (("noabc", "--no_abc"), ("nohist", "--no_abc --no_hist_pool"),
                      ("noge", "--no_abc --no_ge")):
        base_extra = ("--alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2 "
                      "--char_length 0.07 --dpose_threshold 0.055")
        for seed in seeds3:
            s.add("phase4", "pbrsE", seed, 528,
                  extra_override=f"{base_extra} {flag}", name_suffix=f"_{tag}")

    # Phase 4 (ABC-on): E with ABC enabled (beta=0.5) — tests the clip-saturation
    # claim on the T-block d_pose model. NOTE: no --no_abc → ABC active.
    abc_extra = ("--alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2 "
                 "--char_length 0.07 --dpose_threshold 0.055")
    for seed in seeds3:
        s.add("phase4", "pbrsE", seed, 528, extra_override=abc_extra, name_suffix="_abc")

    # Phase 4 positive control (E4-PC): ABC-on on the easy/symmetric disc
    # (position-only d_pose). If ABC helps anywhere it should help here — rules
    # out "the BC was broken" for the negative ABC-on-E result.
    disc_abc_extra = ("--alice_pushes 5 --bob_pushes 10 --max_goals_per_episode 2 "
                      "--char_length 0.0 --dpose_threshold 0.05")
    for seed in seeds3:
        s.add("phase4_pc", "discF", seed, 528, extra_override=disc_abc_extra, name_suffix="_abc")

    # Phase 5a (E6): PBRS grid on pbrsA @528, seed 42
    for k_p in (10, 20, 30, 50):
        for w in (5, 10, 20):
            extra = f"--k_p {k_p} --k_r 5 --w_pos {w} --w_rot {w}"
            s.add("phase5_grid", "pbrsA", 42, 528,
                  extra_override=extra, name_suffix=f"_kp{k_p}_w{w}")

    # Phase 5b (E3): gym-pusht batch crossover, seed 42
    gym_scripts = {"gymA": "train_a_gym_pbrs_simple.py",
                   "gymB": "train_b_gym_pbrs_curriculum.py",
                   "gymC": "train_c_gym_pbrs_asp.py"}
    for name, script in gym_scripts.items():
        for envs in (64, 256):
            for pns in (15, 45):
                exp = f"{name}_e{envs}_p{pns}_s42"
                s.add_gym("phase5_gym", script, 42, envs, pns, exp)

    s.write()


if __name__ == "__main__":
    build()
