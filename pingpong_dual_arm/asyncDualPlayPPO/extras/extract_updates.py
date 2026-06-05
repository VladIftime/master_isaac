import os
import re
import argparse
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Extract and plot metrics from async dual play PPO logs."
    )
    parser.add_argument(
        "--file",
        type=str,
        default="slurm-28114045-med.out",
        help="Log file name in logs/ dir",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    log_file = os.path.join(base_dir, "logs", args.file)
    output_img = os.path.join(base_dir, "logs", args.file + "_plots.png")

    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    # Data structures
    iterations = []
    alice_loss = []
    alice_val = []
    alice_rew = []

    bob_loss = []
    bob_val = []
    bob_rew = []
    bob_abc = []
    bob_sr = []

    alice_valid_goal_rate = []

    # temporary counters for the current iteration
    valid_goals_curr = 0
    invalid_goals_curr = 0

    alice_pattern = re.compile(
        r"\[Alice Update (\d+)\] Loss: ([\-\.\d]+) \| Val: ([\-\.\d]+) \| Rew: ([\-\.\d]+)"
    )
    bob_pattern = re.compile(
        r"\[Bob Update (\d+)\] Loss: ([\-\.\d]+) \| Val: ([\-\.\d]+) \| Rew: ([\-\.\d]+) \| ABC: ([\-\.\d]+) \| SR: ([\-\.\d]+)"
    )
    valid_goal_pattern = re.compile(r"Alice→Bob")
    invalid_goal_pattern = re.compile(r"Alice Invalid Goal")

    with open(log_file, "r") as f:
        for line in f:
            if valid_goal_pattern.search(line):
                valid_goals_curr += 1
            elif invalid_goal_pattern.search(line):
                invalid_goals_curr += 1

            a_match = alice_pattern.search(line)
            if a_match:
                it = int(a_match.group(1))
                iterations.append(it)
                alice_loss.append(float(a_match.group(2)))
                alice_val.append(float(a_match.group(3)))
                alice_rew.append(float(a_match.group(4)))

                # Compute valid goal setup rate for this iteration up to this point
                total_goals = valid_goals_curr + invalid_goals_curr
                rate = valid_goals_curr / total_goals if total_goals > 0 else 0
                alice_valid_goal_rate.append(rate)

                # reset counters for next iteration
                valid_goals_curr = 0
                invalid_goals_curr = 0
                continue

            b_match = bob_pattern.search(line)
            if b_match:
                bob_loss.append(float(b_match.group(2)))
                bob_val.append(float(b_match.group(3)))
                bob_rew.append(float(b_match.group(4)))
                bob_abc.append(float(b_match.group(5)))
                bob_sr.append(float(b_match.group(6)))

    if len(iterations) == 0:
        print("No update logs found.")
        return

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    # 1. Alice Metrics
    axs[0, 0].plot(iterations, alice_loss, label="Loss")
    axs[0, 0].plot(iterations, alice_val, label="Value")
    axs[0, 0].set_title("Alice PPO Metrics")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Alice Reward
    axs[0, 1].plot(iterations, alice_rew, label="Reward", color="green")
    axs[0, 1].set_title("Alice Reward")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Bob Metrics
    plot_iters_bob = iterations[: len(bob_loss)]
    axs[1, 0].plot(plot_iters_bob, bob_loss, label="Loss")
    axs[1, 0].plot(plot_iters_bob, bob_val, label="Value")
    axs[1, 0].plot(plot_iters_bob, bob_abc, label="ABC Loss")
    axs[1, 0].set_title("Bob PPO Metrics")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Bob Reward
    axs[1, 1].plot(plot_iters_bob, bob_rew, label="Reward", color="green")
    axs[1, 1].set_title("Bob Reward")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # 5. Bob Success Rate
    axs[2, 0].plot(plot_iters_bob, bob_sr, label="Success Rate", color="purple")
    axs[2, 0].set_title("Bob Success Rate")
    axs[2, 0].set_ylim([-0.05, 1.05])
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # 6. Alice Valid Goal Rate
    axs[2, 1].plot(
        iterations, alice_valid_goal_rate, label="Valid Goal Rate", color="orange"
    )
    axs[2, 1].set_title("Alice Valid Goal Ratio")
    axs[2, 1].set_ylim([-0.05, 1.05])
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Plot saved to {output_img}")


if __name__ == "__main__":
    main()
