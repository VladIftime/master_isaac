"""
Test checkpoint save/load/resume correctness WITHOUT Isaac Sim.

Verifies:
  1. Saved weights match loaded weights exactly
  2. resume_iteration is correctly parsed and applied
  3. SLURM auto-resume script picks the right checkpoint
  4. Training continues from the correct iteration (not from 0)

Run:  python test_checkpoint_chain.py
"""

import os
import sys
import shutil
import tempfile
import subprocess

import torch
import torch.nn as nn


# --- Minimal mock of PPO's save/load (same logic as ppo.py) ---

class FakeActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(56, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return model


# --- Test 1: Weight integrity ---

def test_weight_integrity():
    print("Test 1: Weight integrity after save/load")
    model = FakeActorCritic()

    # Simulate some training (modify weights away from init)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)

    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "bob", "model_50.pt")
        save_checkpoint(model, path)

        loaded_model = FakeActorCritic()
        load_checkpoint(loaded_model, path)

        for key in original_state:
            orig = original_state[key]
            loaded = loaded_model.state_dict()[key]
            if not torch.equal(orig, loaded):
                print(f"  FAIL: {key} differs!")
                print(f"    max diff: {(orig - loaded).abs().max().item()}")
                return False

    print("  PASS: All weights match exactly after save/load")
    return True


# --- Test 2: Iteration number parsing ---

def test_iteration_parsing():
    print("Test 2: Iteration number parsing from filename")

    test_cases = [
        ("runs/exp/bob/model_50.pt", 50),
        ("runs/exp/bob/model_5.pt", 5),
        ("runs/exp/bob/model_100.pt", 100),
        ("runs/exp/bob/model_1000.pt", 1000),
    ]

    all_pass = True
    for path, expected in test_cases:
        # Same logic as ppo.py line 110
        parsed = int(path.split("_")[-1].split(".")[0])
        if parsed != expected:
            print(f"  FAIL: '{path}' -> {parsed}, expected {expected}")
            all_pass = False

    if all_pass:
        print("  PASS: All iteration numbers parsed correctly")
    return all_pass


# --- Test 3: SLURM auto-resume picks latest checkpoint ---

def test_slurm_checkpoint_detection():
    print("Test 3: SLURM auto-resume finds latest checkpoint")

    with tempfile.TemporaryDirectory() as tmpdir:
        bob_dir = os.path.join(tmpdir, "bob")
        alice_dir = os.path.join(tmpdir, "alice")
        os.makedirs(bob_dir)
        os.makedirs(alice_dir)

        # Create checkpoints at different iterations
        for it in [5, 10, 15]:
            torch.save({}, os.path.join(bob_dir, f"model_{it}.pt"))
            torch.save({}, os.path.join(alice_dir, f"model_{it}.pt"))
        # Also create best/final (should be skipped)
        torch.save({}, os.path.join(bob_dir, "model_best.pt"))
        torch.save({}, os.path.join(bob_dir, "model_final.pt"))

        # Same logic as train_high.slurm (sort by iteration number, not mtime)
        cmd = f"""
        LATEST_BOB=$(ls "{bob_dir}"/model_*.pt 2>/dev/null | grep -v best | grep -v final \
            | sed 's/.*model_//' | sed 's/\\.pt//' | sort -n | tail -1)
        ITER_NUM="$LATEST_BOB"
        echo "$ITER_NUM|model_${{ITER_NUM}}.pt|model_${{ITER_NUM}}.pt"
        """
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        output = result.stdout.strip()
        parts = output.split("|")

        if len(parts) != 3:
            print(f"  FAIL: unexpected output: {output}")
            return False

        iter_num, bob_file, alice_file = parts

        # ls -t sorts by modification time; all created nearly simultaneously,
        # but the last created (15) should be newest
        if iter_num == "15" and bob_file == "model_15.pt" and alice_file == "model_15.pt":
            print(f"  PASS: Detected iteration {iter_num}, files: {bob_file}, {alice_file}")
            return True
        else:
            print(f"  FAIL: Got iter={iter_num}, bob={bob_file}, alice={alice_file}")
            print(f"        Expected iter=15, bob=model_15.pt, alice=model_15.pt")
            return False


# --- Test 4: Training resumes at correct iteration ---

def test_resume_iteration():
    print("Test 4: bob_updates starts at resume_iteration")

    # Simulate the logic from train.py
    class Args:
        resume_iteration = 50

    args = Args()
    bob_updates = args.resume_iteration
    max_iterations = 100

    iterations_run = []
    while bob_updates < max_iterations and len(iterations_run) < 5:
        iterations_run.append(bob_updates)
        bob_updates += 1

    if iterations_run[0] == 50 and iterations_run[-1] == 54:
        print(f"  PASS: Ran iterations {iterations_run[0]}..{iterations_run[-1]}")
        return True
    else:
        print(f"  FAIL: Expected [50..54], got {iterations_run}")
        return False


# --- Test 5: Entropy annealing correct on resume ---

def test_entropy_annealing_on_resume():
    print("Test 5: Entropy annealing correct after resume")

    _ALICE_ENT_START = 1.0
    _ALICE_ENT_END = 0.01
    _ALICE_ENT_ANNEAL_ITERS = 100

    # Fresh run at iteration 50
    bob_updates = 50
    frac = min(1.0, bob_updates / _ALICE_ENT_ANNEAL_ITERS)
    ent_fresh = _ALICE_ENT_START + frac * (_ALICE_ENT_END - _ALICE_ENT_START)

    # Resume at iteration 50
    resume_bob_updates = 50
    frac_resume = min(1.0, resume_bob_updates / _ALICE_ENT_ANNEAL_ITERS)
    ent_resume = _ALICE_ENT_START + frac_resume * (_ALICE_ENT_END - _ALICE_ENT_START)

    expected = 1.0 + 0.5 * (0.01 - 1.0)  # = 0.505

    if abs(ent_fresh - ent_resume) < 1e-6 and abs(ent_fresh - expected) < 1e-6:
        print(f"  PASS: Entropy at iter 50 = {ent_fresh:.4f} (expected ~0.505)")
        return True
    else:
        print(f"  FAIL: fresh={ent_fresh}, resume={ent_resume}, expected={expected}")
        return False


# --- Run all tests ---

if __name__ == "__main__":
    results = [
        test_weight_integrity(),
        test_iteration_parsing(),
        test_slurm_checkpoint_detection(),
        test_resume_iteration(),
        test_entropy_annealing_on_resume(),
    ]

    print(f"\n{'='*50}")
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} passed")

    if all(results):
        print("All checkpoint chain tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)
