# Experiments

Document generated on 2026-04-13 from all available TXT, CSV, and notebook sources under this directory tree.

## 1. Source Corpus and Audit Scope

The analysis in this file was compiled from every available source artifact with extensions .txt, .csv, and .ipynb in the parent folder and sub-folders. The inventory below provides file-level provenance (size and line count).

| file | type | size_bytes | line_count |
| --- | --- | --- | --- |
| 3Dresnet18(16 frames, 24 batch size, 1 epoch).txt | .txt | 24832 | 441 |
| 3Dresnet18(32 frames, 40 batch size, 1 epoch).txt | .txt | 24276 | 430 |
| 3Dresnet18(8 frames, 40 batch size, 1 epoch).txt | .txt | 24316 | 432 |
| R2+1d_18, (16 frames, 24 batchsize, 2 epochs).txt | .txt | 26937 | 467 |
| R2+1d_18, (8 frames, 40 batchsize, 2 epochs).txt | .txt | 25673 | 449 |
| mc3_18(24 batchsize, 16 frames, 2 epochs).txt | .txt | 26368 | 466 |
| mc3_18(32 frames, 40 batch_size, 2 epoch).txt | .txt | 28510 | 526 |
| mc3_18(40 batchsize, 8 frames, 2 epochs).txt | .txt | 25329 | 451 |
| plots.ipynb | .ipynb | 4145915 | 3962 |
| proposed_tables/appendix_per_class_accuracy.csv | .csv | 3433 | 51 |
| proposed_tables/main_adaptation_metrics.csv | .csv | 202 | 4 |
| proposed_tables/main_experimental_setup.csv | .csv | 709 | 10 |
| proposed_tables/main_frame_ablation.csv | .csv | 176 | 4 |
| proposed_tables/main_results.csv | .csv | 895 | 10 |
| r2plus1d_18, (32 frames, 40 batch_size, 2 epochs)).txt | .txt | 25370 | 453 |

## 2. Experimental Setup (Extracted from Logs + Setup CSV)

### 2.1 Dataset and Split

- Number of action classes: 50
- Stratified split: 70-30 (consistently reported in logs and setup CSV)
- Total training samples (summed from class-distribution block): 4676
- Total test samples (summed from class-distribution block): 2005
- Test support range per class: min=30, max=59

Class-distribution table parsed from the logs:

| class_name | train_count | test_count |
| --- | --- | --- |
| BaseballPitch | 105 | 45 |
| Basketball | 96 | 41 |
| BenchPress | 112 | 48 |
| Biking | 101 | 44 |
| Billiards | 105 | 45 |
| BreastStroke | 71 | 30 |
| CleanAndJerk | 78 | 34 |
| Diving | 107 | 46 |
| Drumming | 113 | 48 |
| Fencing | 78 | 33 |
| GolfSwing | 99 | 43 |
| HighJump | 86 | 37 |
| HorseRace | 89 | 38 |
| HorseRiding | 138 | 59 |
| HulaHoop | 87 | 38 |
| JavelinThrow | 82 | 35 |
| JugglingBalls | 85 | 37 |
| JumpRope | 103 | 45 |
| JumpingJack | 86 | 37 |
| Kayaking | 110 | 47 |
| Lunges | 99 | 42 |
| MilitaryParade | 89 | 38 |
| Mixing | 99 | 42 |
| Nunchucks | 105 | 45 |
| PizzaTossing | 80 | 34 |
| PlayingGuitar | 112 | 48 |
| PlayingPiano | 73 | 32 |
| PlayingTabla | 87 | 37 |
| PlayingViolin | 70 | 30 |
| PoleVault | 112 | 48 |
| PommelHorse | 86 | 37 |
| PullUps | 84 | 36 |
| Punch | 112 | 48 |
| PushUps | 74 | 32 |
| RockClimbingIndoor | 104 | 44 |
| RopeClimbing | 91 | 39 |
| Rowing | 96 | 41 |
| SalsaSpin | 93 | 40 |
| SkateBoarding | 84 | 36 |
| Skiing | 101 | 43 |
| Skijet | 70 | 30 |
| SoccerJuggling | 109 | 47 |
| Swing | 96 | 41 |
| TaiChi | 70 | 30 |
| TennisSwing | 117 | 50 |
| ThrowDiscus | 92 | 39 |
| TrampolineJumping | 83 | 36 |
| VolleyballSpiking | 81 | 35 |
| WalkingWithDog | 86 | 37 |
| YoYo | 90 | 38 |

### 2.2 Model/Run Configuration Matrix

| run_id | architecture | frames | batch | epochs | trainable_params | train_test_split | vitta_on | rmga_on |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r3d_18-F8-B40-E1 | 3DResNet18 (r3d_18) | 8 | 40 | 1 | 33,191,922 | 70-30 | True | True |
| r3d_18-F16-B24-E1 | 3DResNet18 (r3d_18) | 16 | 24 | 1 | 33,191,922 | 70-30 | True | True |
| r3d_18-F32-B40-E1 | 3DResNet18 (r3d_18) | 32 | 40 | 1 | 33,191,922 | 70-30 | True | True |
| mc3_18-F8-B40-E2 | MC3-18 | 8 | 40 | 2 | 11,515,890 | 70-30 | True | True |
| mc3_18-F16-B24-E2 | MC3-18 | 16 | 24 | 2 | 11,515,890 | 70-30 | True | True |
| mc3_18-F32-B40-E2 | MC3-18 | 32 | 40 | 2 | 11,515,890 | 70-30 | True | True |
| r2plus1d_18-F8-B40-E2 | R(2+1)D-18 | 8 | 40 | 2 | 31,325,775 | 70-30 | True | True |
| r2plus1d_18-F16-B24-E2 | R(2+1)D-18 | 16 | 24 | 2 | 31,325,775 | 70-30 | True | True |
| r2plus1d_18-F32-B40-E2 | R(2+1)D-18 | 32 | 40 | 2 | 31,325,775 | 70-30 | True | True |

### 2.3 Notebook Pipeline Structure

- Notebook file: plots.ipynb
- Total cells: 37 (code=31, markdown=6)
- Primary markdown sections: Proposed, Required, professional
- Output directories used by the notebook: required_figures_svg, professional_figures_svg, proposed_figures_svg, proposed_tables

## 3. Training Dynamics and Log-Level Telemetry

| run_id | file_name | line_count | train_step_points | vitta_progress_points | rmga_progress_points | class_distribution_rows |
| --- | --- | --- | --- | --- | --- | --- |
| mc3_18-F16-B24-E2 | mc3_18(24 batchsize, 16 frames, 2 epochs).txt | 466 | 40 | 101 | 101 | 50 |
| mc3_18-F32-B40-E2 | mc3_18(32 frames, 40 batch_size, 2 epoch).txt | 526 | 24 | 101 | 87 | 100 |
| mc3_18-F8-B40-E2 | mc3_18(40 batchsize, 8 frames, 2 epochs).txt | 451 | 24 | 101 | 101 | 50 |
| r2plus1d_18-F16-B24-E2 | R2+1d_18, (16 frames, 24 batchsize, 2 epochs).txt | 467 | 40 | 101 | 101 | 50 |
| r2plus1d_18-F32-B40-E2 | r2plus1d_18, (32 frames, 40 batch_size, 2 epochs)).txt | 453 | 24 | 101 | 101 | 50 |
| r2plus1d_18-F8-B40-E2 | R2+1d_18, (8 frames, 40 batchsize, 2 epochs).txt | 449 | 23 | 101 | 101 | 50 |
| r3d_18-F16-B24-E1 | 3Dresnet18(16 frames, 24 batch size, 1 epoch).txt | 441 | 20 | 101 | 101 | 50 |
| r3d_18-F32-B40-E1 | 3Dresnet18(32 frames, 40 batch size, 1 epoch).txt | 430 | 12 | 101 | 101 | 50 |
| r3d_18-F8-B40-E1 | 3Dresnet18(8 frames, 40 batch size, 1 epoch).txt | 432 | 12 | 101 | 101 | 50 |

Per-epoch summaries parsed from log files:

| run_id | file_name | epoch | train_loss | train_acc | test_loss | test_acc |
| --- | --- | --- | --- | --- | --- | --- |
| mc3_18-F16-B24-E2 | mc3_18(24 batchsize, 16 frames, 2 epochs).txt | 1 | 1.986 | 72.140 | 2.108 | 64.690 |
| mc3_18-F16-B24-E2 | mc3_18(24 batchsize, 16 frames, 2 epochs).txt | 2 | 0.986 | 96.970 | 2.040 | 61.350 |
| mc3_18-F32-B40-E2 | mc3_18(32 frames, 40 batch_size, 2 epoch).txt | 1 | 2.151 | 68.920 | 2.304 | 58.350 |
| mc3_18-F32-B40-E2 | mc3_18(32 frames, 40 batch_size, 2 epoch).txt | 2 | 1.051 | 96.250 | 2.117 | 62.000 |
| mc3_18-F8-B40-E2 | mc3_18(40 batchsize, 8 frames, 2 epochs).txt | 1 | 2.207 | 67.520 | 2.361 | 56.210 |
| mc3_18-F8-B40-E2 | mc3_18(40 batchsize, 8 frames, 2 epochs).txt | 2 | 1.089 | 95.620 | 2.249 | 57.110 |
| r2plus1d_18-F16-B24-E2 | R2+1d_18, (16 frames, 24 batchsize, 2 epochs).txt | 1 | 1.798 | 75.430 | 2.288 | 51.870 |
| r2plus1d_18-F16-B24-E2 | R2+1d_18, (16 frames, 24 batchsize, 2 epochs).txt | 2 | 0.867 | 97.790 | 2.112 | 57.760 |
| r2plus1d_18-F32-B40-E2 | r2plus1d_18, (32 frames, 40 batch_size, 2 epochs)).txt | 1 | 1.863 | 74.610 | 2.284 | 54.910 |
| r2plus1d_18-F32-B40-E2 | r2plus1d_18, (32 frames, 40 batch_size, 2 epochs)).txt | 2 | 0.852 | 98.280 | 2.009 | 61.250 |
| r2plus1d_18-F8-B40-E2 | R2+1d_18, (8 frames, 40 batchsize, 2 epochs).txt | 1 | 2.149 | 66.640 | 2.446 | 50.670 |
| r2plus1d_18-F8-B40-E2 | R2+1d_18, (8 frames, 40 batchsize, 2 epochs).txt | 2 | 0.984 | 95.450 | 2.467 | 47.780 |
| r3d_18-F16-B24-E1 | 3Dresnet18(16 frames, 24 batch size, 1 epoch).txt | 1 | 1.880 | 72.830 | 2.461 | 44.540 |
| r3d_18-F32-B40-E1 | 3Dresnet18(32 frames, 40 batch size, 1 epoch).txt | 1 | 2.027 | 70.800 | 2.496 | 43.640 |
| r3d_18-F8-B40-E1 | 3Dresnet18(8 frames, 40 batch size, 1 epoch).txt | 1 | 2.328 | 60.730 | 2.780 | 37.460 |

## 4. Main Quantitative Results

### 4.1 Run-Level Results (CSV + Log-Enriched Metrics)

| run_id | architecture | frames | batch | best_epoch | best_train_loss | best_test_loss | baseline_acc | vitta_acc | rmga_acc | vitta_gain | rmga_gain | rmga_minus_vitta | vitta_entropy | rmga_entropy | rmga_conf | stability_index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r3d_18-F8-B40-E1 | 3DResNet18 (r3d_18) | 8 | 40 | 1 | 2.328 | 2.780 | 37.460 | 45.790 | 45.340 | 8.330 | 7.880 | -0.450 | 3.476 | 3.250 | 0.227 | 25.860 |
| r3d_18-F16-B24-E1 | 3DResNet18 (r3d_18) | 16 | 24 | 1 | 1.880 | 2.461 | 44.540 | 50.920 | 48.630 | 6.380 | 4.090 | -2.290 | 3.329 | 2.973 | 0.293 | 27.740 |
| r3d_18-F32-B40-E1 | 3DResNet18 (r3d_18) | 32 | 40 | 1 | 2.027 | 2.496 | 43.640 | 50.370 | 48.330 | 6.730 | 4.690 | -2.040 | 3.512 | 3.239 | 0.230 | 29.930 |
| mc3_18-F8-B40-E2 | MC3-18 | 8 | 40 | 2 | 1.089 | 2.249 | 57.110 | 74.060 | 67.680 | 16.950 | 10.570 | -6.380 | 3.590 | 3.286 | 0.232 | 17.680 |
| mc3_18-F16-B24-E2 | MC3-18 | 16 | 24 | 1 | 1.986 | 2.108 | 64.690 | 76.360 | 73.220 | 11.670 | 8.530 | -3.140 | 3.575 | 3.260 | 0.239 | 18.110 |
| mc3_18-F32-B40-E2 | MC3-18 | 32 | 40 | 2 | 1.051 | 2.117 | 62.000 | 78.050 | 71.370 | 16.050 | 9.370 | -6.680 | 3.603 | 3.296 | 0.229 | 16.700 |
| r2plus1d_18-F8-B40-E2 | R(2+1)D-18 | 8 | 40 | 1 | 2.149 | 2.446 | 50.670 | 64.940 | 63.240 | 14.270 | 12.570 | -1.700 | 3.431 | 3.124 | 0.276 | 21.960 |
| r2plus1d_18-F16-B24-E2 | R(2+1)D-18 | 16 | 24 | 2 | 0.867 | 2.112 | 57.760 | 73.570 | 70.170 | 15.810 | 12.410 | -3.400 | 3.205 | 2.758 | 0.369 | 21.470 |
| r2plus1d_18-F32-B40-E2 | R(2+1)D-18 | 32 | 40 | 2 | 0.852 | 2.009 | 61.250 | 72.820 | 70.320 | 11.570 | 9.070 | -2.500 | 3.256 | 2.853 | 0.350 | 21.910 |

### 4.2 Global Aggregates Across 9 Runs

- Baseline mean ± std: 53.236 ± 9.558
- ViTTA mean ± std: 65.209 ± 12.733
- RMGA mean ± std: 62.033 ± 11.328
- Mean ViTTA gain vs baseline: 11.973
- Mean RMGA gain vs baseline: 8.798
- Mean RMGA - ViTTA delta: -3.176

### 4.3 Architecture-Wise Averages

| architecture | baseline_acc | vitta_acc | rmga_acc | vitta_gain | rmga_gain | rmga_minus_vitta |
| --- | --- | --- | --- | --- | --- | --- |
| 3DResNet18 (r3d_18) | 41.880 | 49.027 | 47.433 | 7.147 | 5.553 | -1.593 |
| MC3-18 | 61.267 | 76.157 | 70.757 | 14.890 | 9.490 | -5.400 |
| R(2+1)D-18 | 56.560 | 70.443 | 67.910 | 13.883 | 11.350 | -2.533 |

### 4.4 Best Baseline Configuration Per Architecture

| architecture | run_id | frames | batch | baseline_acc | vitta_acc | rmga_acc |
| --- | --- | --- | --- | --- | --- | --- |
| 3DResNet18 (r3d_18) | r3d_18-F16-B24-E1 | 16 | 24 | 44.540 | 50.920 | 48.630 |
| MC3-18 | mc3_18-F16-B24-E2 | 16 | 24 | 64.690 | 76.360 | 73.220 |
| R(2+1)D-18 | r2plus1d_18-F32-B40-E2 | 32 | 40 | 61.250 | 72.820 | 70.320 |

## 5. Temporal Ablation

Frame-ablation table from proposed_tables/main_frame_ablation.csv:

| architecture | 8_frames | 16_frames | 32_frames | best_setting |
| --- | --- | --- | --- | --- |
| 3DResNet18 (r3d_18) | 37.460 | 44.540 | 43.640 | 16_frames |
| MC3-18 | 57.110 | 64.690 | 62.000 | 16_frames |
| R(2+1)D-18 | 50.670 | 57.760 | 61.250 | 32_frames |

Derived baseline deltas from 8->16->32 frames:

| architecture | 8_frames | 16_frames | 32_frames | delta_8_to_16 | delta_16_to_32 | delta_8_to_32 |
| --- | --- | --- | --- | --- | --- | --- |
| 3DResNet18 (r3d_18) | 37.460 | 44.540 | 43.640 | 7.080 | -0.900 | 6.180 |
| MC3-18 | 57.110 | 64.690 | 62.000 | 7.580 | -2.690 | 4.890 |
| R(2+1)D-18 | 50.670 | 57.760 | 61.250 | 7.090 | 3.490 | 10.580 |

## 6. Adaptation Analysis

### 6.1 Method-Level Adaptation Metrics (from CSV)

| method | accuracy | mean_entropy | mean_confidence | stability_index |
| --- | --- | --- | --- | --- |
| Baseline | 53.236 | NA | NA | NA |
| ViTTA | 65.209 | 3.442 | NA | NA |
| RMGA | 62.033 | 3.116 | 0.272 | 22.373 |

### 6.2 Winner Count by Adaptation Method (run-wise)

- ViTTA: 9 / 9 runs

### 6.3 Interpretation

- Both adaptation methods consistently outperform baseline at run level, with larger average gains for ViTTA in this corpus.
- RMGA shows lower mean entropy than ViTTA in aggregate, but lower top-1 than ViTTA in all 9 runs under the present settings.
- Stability index is reported in RMGA logs and included in the run-level enriched table for traceability.

## 7. Per-Class Behavior (50-Class Analysis)

- Classes with positive RMGA-ViTTA delta: 14
- Classes with negative RMGA-ViTTA delta: 34
- Classes with zero delta: 2

Top-10 positive deltas (RMGA improvements over ViTTA):

| class_name | support | vitta_acc | rmga_acc | delta |
| --- | --- | --- | --- | --- |
| PullUps | 36 | 59.578 | 73.144 | 13.567 |
| Swing | 41 | 88.878 | 94.044 | 5.167 |
| CleanAndJerk | 34 | 38.544 | 43.144 | 4.600 |
| Billiards | 45 | 80.011 | 84.200 | 4.189 |
| MilitaryParade | 38 | 95.622 | 97.967 | 2.344 |
| Lunges | 42 | 34.389 | 36.244 | 1.856 |
| PoleVault | 48 | 53.933 | 55.556 | 1.622 |
| JumpRope | 45 | 63.700 | 64.933 | 1.233 |
| YoYo | 38 | 64.033 | 65.211 | 1.178 |
| HulaHoop | 38 | 44.156 | 45.011 | 0.856 |

Top-10 negative deltas (RMGA underperforming ViTTA):

| class_name | support | vitta_acc | rmga_acc | delta |
| --- | --- | --- | --- | --- |
| Fencing | 33 | 67.333 | 53.189 | -14.144 |
| Nunchucks | 45 | 54.311 | 40.244 | -14.067 |
| Rowing | 41 | 65.856 | 53.389 | -12.467 |
| HorseRiding | 59 | 88.867 | 76.456 | -12.411 |
| Skijet | 30 | 41.111 | 31.844 | -9.267 |
| PizzaTossing | 34 | 33.333 | 24.833 | -8.500 |
| PlayingGuitar | 48 | 82.411 | 74.078 | -8.333 |
| Skiing | 43 | 88.611 | 81.133 | -7.478 |
| Kayaking | 47 | 73.278 | 65.956 | -7.322 |
| SoccerJuggling | 47 | 60.278 | 53.189 | -7.089 |

Full 50-class appendix table (from proposed_tables/appendix_per_class_accuracy.csv):

| class_name | support | vitta_acc | rmga_acc | delta |
| --- | --- | --- | --- | --- |
| PullUps | 36 | 59.578 | 73.144 | 13.567 |
| Swing | 41 | 88.878 | 94.044 | 5.167 |
| CleanAndJerk | 34 | 38.544 | 43.144 | 4.600 |
| Billiards | 45 | 80.011 | 84.200 | 4.189 |
| MilitaryParade | 38 | 95.622 | 97.967 | 2.344 |
| Lunges | 42 | 34.389 | 36.244 | 1.856 |
| PoleVault | 48 | 53.933 | 55.556 | 1.622 |
| JumpRope | 45 | 63.700 | 64.933 | 1.233 |
| YoYo | 38 | 64.033 | 65.211 | 1.178 |
| HulaHoop | 38 | 44.156 | 45.011 | 0.856 |
| PlayingPiano | 32 | 32.622 | 33.333 | 0.711 |
| JugglingBalls | 37 | 72.989 | 73.589 | 0.600 |
| PlayingViolin | 30 | 60.744 | 61.122 | 0.378 |
| PlayingTabla | 37 | 94.000 | 94.300 | 0.300 |
| JumpingJack | 37 | 87.700 | 87.700 | 0.000 |
| GolfSwing | 43 | 66.144 | 66.144 | 0.000 |
| Diving | 46 | 85.267 | 84.767 | -0.500 |
| ThrowDiscus | 39 | 88.878 | 88.322 | -0.556 |
| PushUps | 32 | 63.533 | 62.822 | -0.711 |
| RopeClimbing | 39 | 71.233 | 70.389 | -0.844 |
| SkateBoarding | 36 | 55.867 | 54.933 | -0.933 |
| TrampolineJumping | 36 | 79.011 | 76.856 | -2.156 |
| RockClimbingIndoor | 44 | 60.867 | 58.578 | -2.289 |
| HighJump | 37 | 21.922 | 19.500 | -2.422 |
| HorseRace | 38 | 70.178 | 66.378 | -3.800 |
| Biking | 44 | 86.122 | 82.322 | -3.800 |
| WalkingWithDog | 37 | 53.467 | 49.544 | -3.922 |
| Punch | 48 | 71.522 | 67.600 | -3.922 |
| JavelinThrow | 35 | 22.867 | 18.744 | -4.122 |
| TaiChi | 30 | 66.656 | 62.233 | -4.422 |
| SalsaSpin | 40 | 58.889 | 54.444 | -4.444 |
| BreastStroke | 30 | 70.011 | 65.567 | -4.444 |
| TennisSwing | 50 | 52.222 | 47.556 | -4.667 |
| VolleyballSpiking | 35 | 76.833 | 72.056 | -4.778 |
| BenchPress | 48 | 79.856 | 74.778 | -5.078 |
| BaseballPitch | 45 | 46.922 | 41.733 | -5.189 |
| Mixing | 42 | 78.844 | 73.544 | -5.300 |
| Drumming | 48 | 55.778 | 49.767 | -6.011 |
| PommelHorse | 37 | 57.667 | 51.367 | -6.300 |
| Basketball | 41 | 53.933 | 46.889 | -7.044 |
| SoccerJuggling | 47 | 60.278 | 53.189 | -7.089 |
| Kayaking | 47 | 73.278 | 65.956 | -7.322 |
| Skiing | 43 | 88.611 | 81.133 | -7.478 |
| PlayingGuitar | 48 | 82.411 | 74.078 | -8.333 |
| PizzaTossing | 34 | 33.333 | 24.833 | -8.500 |
| Skijet | 30 | 41.111 | 31.844 | -9.267 |
| HorseRiding | 59 | 88.867 | 76.456 | -12.411 |
| Rowing | 41 | 65.856 | 53.389 | -12.467 |
| Nunchucks | 45 | 54.311 | 40.244 | -14.067 |
| Fencing | 33 | 67.333 | 53.189 | -14.144 |

## 8. Cross-File Consistency Checks

| item | observed | expected | status |
| --- | --- | --- | --- |
| TXT run count | 9 | 9 | PASS |
| CSV run count (main_results) | 9 | 9 | PASS |
| CSV run count (setup) | 9 | 9 | PASS |
| Per-class rows | 50 | 50 | PASS |
| Notebook sections present | Proposed, Required, professional | Required/professional/Proposed | PASS |

## 9. Reproducibility Notes

- Log naming convention encodes architecture, frame count, batch size, and epoch budget; these were cross-checked against in-file metadata.
- For logs missing explicit [TRAIN DONE] summary, best baseline accuracy was recovered from epoch summary/new-best checkpoints and validated against CSV outputs.
- All derived statements above are grounded in the source corpus listed in Section 1 and traceable to either raw logs, proposed CSV tables, or notebook-defined computation paths.

## 10. Figure/Table Artifact Coverage from Notebook Execution

The notebook contains Required, professional, and Proposed sections and exports publication-grade SVG figures and CSV tables. Proposed artifacts verified in this workspace include:

- proposed_figures_svg/proposed_01_dataset_composition_stacked.svg
- proposed_figures_svg/proposed_02_training_convergence.svg
- proposed_figures_svg/proposed_03_architecture_comparison_best_settings.svg
- proposed_figures_svg/proposed_04_temporal_ablation.svg
- proposed_figures_svg/proposed_05_adaptation_comparison.svg
- proposed_figures_svg/proposed_06_per_class_performance.svg
- proposed_figures_svg/proposed_07_per_class_improvement_delta.svg
- proposed_tables/appendix_per_class_accuracy.csv
- proposed_tables/main_adaptation_metrics.csv
- proposed_tables/main_experimental_setup.csv
- proposed_tables/main_frame_ablation.csv
- proposed_tables/main_results.csv

