#!/usr/bin/env python3
"""Check Initial vs Final reward distribution in fail data."""

import json
import argparse
from collections import defaultdict, Counter

parser = argparse.ArgumentParser(description="Check initial vs final reward distribution in fail data")
parser.add_argument("--fail-data", default="gui360_full/converted_fail_data.json",
                    help="Path to converted_fail_data.json")
args = parser.parse_args()

with open(args.fail_data, 'r') as f:
    data = json.load(f)

# Build trajectories
trajectories = defaultdict(list)
for sample in data:
    traj_id = sample.get('execution_id', sample['id'].rsplit('_step', 1)[0])
    trajectories[traj_id].append(sample)

# Sort steps
for traj_id in trajectories:
    trajectories[traj_id] = sorted(trajectories[traj_id], key=lambda x: x.get('step_id', 0))

# Multi-step only
multi_step = [steps for steps in trajectories.values() if len(steps) >= 2]
print(f'Multi-step trajectories: {len(multi_step)}')

# Initial vs Final rewards
initial_rewards = [steps[0].get('reward', 0) for steps in multi_step]
final_rewards = [steps[-1].get('reward', 0) for steps in multi_step]

print(f'\nInitial reward distribution:')
for r in sorted(set(initial_rewards)):
    cnt = initial_rewards.count(r)
    print(f'  {r}: {cnt:,} ({100*cnt/len(initial_rewards):.1f}%)')

print(f'\nFinal reward distribution:')
for r in sorted(set(final_rewards)):
    cnt = final_rewards.count(r)
    print(f'  {r}: {cnt:,} ({100*cnt/len(final_rewards):.1f}%)')

# Cross-tabulation: Initial=0.0 vs Final
print(f'\n[Initial=0.0] -> Final distribution:')
init_0_indices = [i for i, r in enumerate(initial_rewards) if r == 0.0]
finals_for_init_0 = [final_rewards[i] for i in init_0_indices]
for r in sorted(set(finals_for_init_0)):
    cnt = finals_for_init_0.count(r)
    print(f'  Final={r}: {cnt:,} ({100*cnt/len(finals_for_init_0):.1f}%)')

# Cross-tabulation: Initial=0.5 vs Final
print(f'\n[Initial=0.5] -> Final distribution:')
init_05_indices = [i for i, r in enumerate(initial_rewards) if r == 0.5]
finals_for_init_05 = [final_rewards[i] for i in init_05_indices]
for r in sorted(set(finals_for_init_05)):
    cnt = finals_for_init_05.count(r)
    print(f'  Final={r}: {cnt:,} ({100*cnt/len(finals_for_init_05):.1f}%)')

# Progression patterns
print(f'\nReward progression patterns:')
decreasing = sum(1 for i in range(len(multi_step)) if initial_rewards[i] > final_rewards[i])
constant = sum(1 for i in range(len(multi_step)) if initial_rewards[i] == final_rewards[i])
increasing = sum(1 for i in range(len(multi_step)) if initial_rewards[i] < final_rewards[i])
print(f'  Decreasing (init>final): {decreasing:,} ({100*decreasing/len(multi_step):.1f}%)')
print(f'  Constant (init==final): {constant:,} ({100*constant/len(multi_step):.1f}%)')
print(f'  Increasing (init<final): {increasing:,} ({100*increasing/len(multi_step):.1f}%)')

print(f'\n=== SUMMARY ===')
print(f'Total multi-step: {len(multi_step):,}')
print(f'Initial 0.0: {initial_rewards.count(0.0):,} ({100*initial_rewards.count(0.0)/len(initial_rewards):.1f}%)')
print(f'Initial 0.5: {initial_rewards.count(0.5):,} ({100*initial_rewards.count(0.5)/len(initial_rewards):.1f}%)')
print(f'Final 0.0: {final_rewards.count(0.0):,} ({100*final_rewards.count(0.0)/len(final_rewards):.1f}%)')
print(f'Final 0.5: {final_rewards.count(0.5):,} ({100*final_rewards.count(0.5)/len(final_rewards):.1f}%)')
