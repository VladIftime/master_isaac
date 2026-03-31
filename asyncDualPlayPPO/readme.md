python tests/test_abc.py --step 1 --num_envs 64 --num_iterations 200 --headless > test_step1.out 2>&1 &
tail -f test_step1.out | grep -E "(Iter|SR|PASSED|FAILED|Error)"
Pass condition: SR > 50% by iter ~100-200

Step 2 — Pure BC (does the ABC loss actually work?)
bash
python tests/test_abc.py --step 2 --num_envs 4 --num_iterations 100 --headless > test_step2.out 2>&1 &
tail -f test_step2.out | grep -E "(Iter|BC Loss|Dim0|PASSED|FAILED|Error)"
Pass condition: BC loss decreases + Dim0 mode → 8
This is the fastest test (~5-10 min) and needs no GPU search — just gradient flow. Run this first.

Step 3 — Full Integration (does the whole ASP loop work?)
bash
python tests/test_abc.py --step 3 --num_envs 64 --num_iterations 200 --headless > test_step3.out 2>&1 &
tail -f test_step3.out | grep -E "(Iter|Alice|Bob SR|PASSED|FAILED|Error)"
Pass condition: Alice valid-goal rate > 10% AND Bob SR > 5%

Recommended order
Step 2 first → Step 1 → Step 3
Step	Tests	Time est.	If it fails...
2	ABC loss math	~5-10 min	BC gradient broken
1	PPO + env	~20-40 min	RMPFlow/reward signal broken
3	Full loop	~30-60 min	Phase transitions broken