# Validation Summary

| Model | SR | Avg Pushes | Avg PosErr | Avg RotErr | Avg Cov | Tests |
|-------|----|-----------|------------|----------|--------|-------|
| A_simp (no curriculum) | 80.0% | 23.5 | 0.032 m | 0.568 rad | 33.3% | 30 |
| B_curr (P82 curriculum) | 76.7% | 26.7 | 0.023 m | 0.663 rad | 26.9% | 30 |
| E aspirant d_pose | 6.7% | 12.8 | 0.143 m | 1.612 rad | 1.4% | 30 |
| F aspirant disc | 6.7% | 11.9 | 0.197 m | 1.576 rad | 2.1% | 30 |
| G tasp d_pose | 16.7% | 12.2 | 0.158 m | 1.457 rad | 5.9% | 30 |
| H tasp disc | 10.0% | 12.6 | 0.186 m | 1.260 rad | 2.0% | 30 |

## By Difficulty

| Model | Easy SR | Medium SR | Hard SR | Easy Pushes | Medium Pushes | Hard Pushes |
|-------|---------|-----------|---------|-------------|---------------|------------|
| A_simp (no curriculum) | 100.0% | 66.7% | 80.0% | 28.8 | 24.2 | 24.0 |
| B_curr (P82 curriculum) | 100.0% | 66.7% | 80.0% | 29.8 | 28.2 | 26.4 |
| E aspirant d_pose | 50.0% | 0.0% | 0.0% | 11.5 | 13.2 | 15.0 |
| F aspirant disc | 50.0% | 0.0% | 0.0% | 11.8 | 12.5 | 13.3 |
| G tasp d_pose | 75.0% | 16.7% | 0.0% | 10.5 | 13.2 | 13.1 |
| H tasp disc | 50.0% | 16.7% | 0.0% | 11.2 | 12.5 | 14.2 |

## By Test Type

| Model | Pos-Only SR | Pos+Rot SR | Pos-Only Pushes | Pos+Rot Pushes |
|-------|-------------|-----------|-----------------|---------------|
| A_simp (no curriculum) | 100.0% | 70.0% | 27.8 | 21.3 |
| B_curr (P82 curriculum) | 100.0% | 65.0% | 28.8 | 25.6 |
| E aspirant d_pose | 20.0% | 0.0% | 14.1 | 12.1 |
| F aspirant disc | 20.0% | 0.0% | 12.6 | 11.6 |
| G tasp d_pose | 30.0% | 10.0% | 13.2 | 11.8 |
| H tasp disc | 30.0% | 0.0% | 13.4 | 12.2 |
