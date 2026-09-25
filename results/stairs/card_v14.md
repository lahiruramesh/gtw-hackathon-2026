# Policy card: `runs/g1-stairs-v14/run/params.pkl`

git `9aa46e7` · trained from `runs/g1-stairs-v12/run/ckpt_00254279680.pkl` · 100M steps · lr 0.0001 · scan `uniform`

**Certified envelope: stairs ≤ 6.4 cm** (0.7 m/s, 1.4 Hz cadence; required: strict, strict_camera) · target 10.0 cm: **NOT YET**

| condition | runs | crossed | fell | fall rate (95 % CI) | certified | torque-saturated | joint at limit |
|---|---|---|---|---|---|---|---|
| strict (required) | 96 | 95 | 1 | 1.0 % (0.2-5.7) | 9.9 cm | 4.6 % | 11.3 % |
| strict_camera (required) | 96 | 86 | 8 | 8.3 % (4.3-15.6) | 6.4 cm | 4.8 % | 13.4 % |
| speed_0.6 | 96 | 89 | 0 | 0.0 % (0.0-3.8) | 11.6 cm | 3.6 % | 10.9 % |
| tread_30cm | 32 | 32 | 0 | 0.0 % (0.0-10.7) | 16.0 cm | 5.0 % | 8.6 % |
| push | 32 | 30 | 0 | 0.0 % (0.0-10.7) | 11.6 cm | 4.5 % | 10.9 % |
| low_friction | 32 | 30 | 2 | 6.2 % (1.7-20.1) | 8.1 cm | 5.5 % | 8.9 % |
| payload_5kg | 32 | 25 | 4 | 12.5 % (5.0-28.1) | 4.7 cm | 2.6 % | 9.0 % |
| delay_20ms | 32 | 8 | 17 | 53.1 % (36.4-69.1) | 3.0 cm | 8.2 % | 23.9 % |

Per step height (crossed/runs, falls):

- **strict**: 3.0 cm 12/12 fell 0, 4.7 cm 12/12 fell 0, 6.4 cm 12/12 fell 0, 8.1 cm 12/12 fell 0, 9.9 cm 12/12 fell 0, 11.6 cm 11/12 fell 1, 13.3 cm 12/12 fell 0, 15.0 cm 12/12 fell 0
- **strict_camera**: 3.0 cm 12/12 fell 0, 4.7 cm 12/12 fell 0, 6.4 cm 12/12 fell 0, 8.1 cm 11/12 fell 0, 9.9 cm 9/12 fell 2, 11.6 cm 11/12 fell 1, 13.3 cm 8/12 fell 4, 15.0 cm 11/12 fell 1
- **speed_0.6**: 3.0 cm 12/12 fell 0, 4.7 cm 12/12 fell 0, 6.4 cm 12/12 fell 0, 8.1 cm 12/12 fell 0, 9.9 cm 12/12 fell 0, 11.6 cm 12/12 fell 0, 13.3 cm 10/12 fell 0, 15.0 cm 7/12 fell 0
- **tread_30cm**: 2.0 cm 4/4 fell 0, 4.0 cm 4/4 fell 0, 6.0 cm 4/4 fell 0, 8.0 cm 4/4 fell 0, 10.0 cm 4/4 fell 0, 12.0 cm 4/4 fell 0, 14.0 cm 4/4 fell 0, 16.0 cm 4/4 fell 0
- **push**: 3.0 cm 4/4 fell 0, 4.7 cm 4/4 fell 0, 6.4 cm 4/4 fell 0, 8.1 cm 4/4 fell 0, 9.9 cm 4/4 fell 0, 11.6 cm 4/4 fell 0, 13.3 cm 3/4 fell 0, 15.0 cm 3/4 fell 0
- **low_friction**: 3.0 cm 4/4 fell 0, 4.7 cm 4/4 fell 0, 6.4 cm 4/4 fell 0, 8.1 cm 4/4 fell 0, 9.9 cm 3/4 fell 1, 11.6 cm 4/4 fell 0, 13.3 cm 3/4 fell 1, 15.0 cm 4/4 fell 0
- **payload_5kg**: 3.0 cm 4/4 fell 0, 4.7 cm 4/4 fell 0, 6.4 cm 3/4 fell 0, 8.1 cm 3/4 fell 1, 9.9 cm 4/4 fell 0, 11.6 cm 3/4 fell 1, 13.3 cm 2/4 fell 0, 15.0 cm 2/4 fell 2
- **delay_20ms**: 3.0 cm 4/4 fell 0, 4.7 cm 2/4 fell 1, 6.4 cm 1/4 fell 1, 8.1 cm 1/4 fell 1, 9.9 cm 0/4 fell 4, 11.6 cm 0/4 fell 3, 13.3 cm 0/4 fell 4, 15.0 cm 0/4 fell 3
