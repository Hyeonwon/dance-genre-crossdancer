# 장르별 중요 특징 상위 10개 (XGBoost, 처음 보는 댄서 테스트 윈도우의 평균 |SHAP|)

정답 장르 점수에 대한 기여 크기. 괄호 안은 신체 부위.

## Break (gBR)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | root_height_mean | global | 0.990 |
| 2 | motion_volume | global | 0.623 |
| 3 | r_ankle_speed_mean | legs | 0.589 |
| 4 | r_elbow_jerk_mean | arms | 0.576 |
| 5 | l_ankle_speed_mean | legs | 0.547 |
| 6 | r_wrist_speed_mean | arms | 0.483 |
| 7 | l_wrist_speed_mean | arms | 0.425 |
| 8 | r_knee_speed_max | legs | 0.238 |
| 9 | r_elbow_speed_max | arms | 0.188 |
| 10 | r_ankle_speed_std | legs | 0.187 |

## Pop (gPO)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | root_height_range | global | 1.895 |
| 2 | root_height_mean | global | 1.637 |
| 3 | lr_symmetry | global | 0.345 |
| 4 | r_shoulder_speed_max | arms | 0.241 |
| 5 | l_ear_speed_mean | head | 0.225 |
| 6 | l_wrist_jerk_mean | arms | 0.193 |
| 7 | hspeed_mean | global | 0.181 |
| 8 | nose_acc_mean | head | 0.180 |
| 9 | r_elbow_speed_std | arms | 0.166 |
| 10 | motion_volume | global | 0.163 |

## Lock (gLO)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | root_height_mean | global | 2.631 |
| 2 | hspeed_mean | global | 0.483 |
| 3 | l_knee_speed_mean | legs | 0.463 |
| 4 | nose_acc_mean | head | 0.410 |
| 5 | r_ankle_jerk_mean | legs | 0.388 |
| 6 | l_hip_speed_max | legs | 0.318 |
| 7 | lr_symmetry | global | 0.279 |
| 8 | root_height_range | global | 0.233 |
| 9 | r_elbow_speed_std | arms | 0.205 |
| 10 | r_ear_jerk_mean | head | 0.200 |

## Waack (gWA)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | root_height_mean | global | 1.751 |
| 2 | r_ankle_jerk_mean | legs | 0.546 |
| 3 | l_elbow_jerk_mean | arms | 0.463 |
| 4 | r_ankle_speed_mean | legs | 0.404 |
| 5 | root_height_range | global | 0.402 |
| 6 | l_elbow_speed_max | arms | 0.308 |
| 7 | r_ankle_speed_std | legs | 0.294 |
| 8 | l_elbow_speed_mean | arms | 0.279 |
| 9 | r_shoulder_speed_std | arms | 0.252 |
| 10 | upper_lower_energy_ratio | global | 0.238 |

## Middle Hip-hop (gMH)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | l_eye_jerk_mean | head | 0.860 |
| 2 | root_height_mean | global | 0.557 |
| 3 | rhythm_freq_hz | global | 0.401 |
| 4 | r_knee_speed_mean | legs | 0.388 |
| 5 | motion_volume | global | 0.381 |
| 6 | l_knee_jerk_mean | legs | 0.381 |
| 7 | r_ankle_speed_mean | legs | 0.364 |
| 8 | r_knee_jerk_mean | legs | 0.325 |
| 9 | r_wrist_speed_std | arms | 0.320 |
| 10 | l_ankle_speed_mean | legs | 0.279 |

## LA Hip-hop (gLH)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | root_height_mean | global | 1.484 |
| 2 | root_height_range | global | 0.518 |
| 3 | l_wrist_jerk_mean | arms | 0.376 |
| 4 | l_ear_jerk_mean | head | 0.358 |
| 5 | r_ear_jerk_mean | head | 0.313 |
| 6 | r_wrist_jerk_mean | arms | 0.309 |
| 7 | l_eye_jerk_mean | head | 0.272 |
| 8 | r_elbow_jerk_mean | arms | 0.244 |
| 9 | l_hip_speed_mean | legs | 0.235 |
| 10 | motion_volume | global | 0.220 |

## House (gHO)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | root_height_mean | global | 0.836 |
| 2 | l_hip_speed_mean | legs | 0.782 |
| 3 | lr_symmetry | global | 0.696 |
| 4 | upper_lower_energy_ratio | global | 0.648 |
| 5 | l_ear_jerk_mean | head | 0.430 |
| 6 | motion_volume | global | 0.369 |
| 7 | l_wrist_jerk_mean | arms | 0.312 |
| 8 | root_height_range | global | 0.279 |
| 9 | l_shoulder_speed_mean | arms | 0.257 |
| 10 | rhythm_freq_hz | global | 0.243 |

## Krump (gKR)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | lr_symmetry | global | 0.616 |
| 2 | root_height_mean | global | 0.503 |
| 3 | l_eye_speed_mean | head | 0.477 |
| 4 | r_elbow_speed_std | arms | 0.475 |
| 5 | l_wrist_speed_std | arms | 0.407 |
| 6 | r_eye_speed_mean | head | 0.384 |
| 7 | hspeed_mean | global | 0.338 |
| 8 | r_wrist_speed_std | arms | 0.319 |
| 9 | l_hip_speed_mean | legs | 0.303 |
| 10 | r_knee_speed_std | legs | 0.275 |

## Street Jazz (gJS)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | r_eye_speed_mean | head | 0.570 |
| 2 | root_height_range | global | 0.489 |
| 3 | root_height_mean | global | 0.446 |
| 4 | r_knee_speed_mean | legs | 0.419 |
| 5 | nose_speed_mean | head | 0.418 |
| 6 | l_knee_speed_mean | legs | 0.387 |
| 7 | l_elbow_speed_mean | arms | 0.363 |
| 8 | r_knee_acc_mean | legs | 0.299 |
| 9 | motion_volume | global | 0.295 |
| 10 | l_eye_speed_mean | head | 0.285 |

## Ballet Jazz (gJB)

| 순위 | 특징 | 부위 | 평균 \|SHAP\| |
|---|---|---|---|
| 1 | motion_volume | global | 0.973 |
| 2 | root_height_mean | global | 0.822 |
| 3 | hspeed_mean | global | 0.521 |
| 4 | l_hip_speed_mean | legs | 0.437 |
| 5 | l_hip_speed_std | legs | 0.430 |
| 6 | l_hip_jerk_mean | legs | 0.331 |
| 7 | l_wrist_speed_std | arms | 0.317 |
| 8 | r_shoulder_jerk_mean | arms | 0.292 |
| 9 | l_shoulder_speed_mean | arms | 0.280 |
| 10 | root_height_range | global | 0.262 |
