# Policy card: `runs/g1-stairs-v12/run/ckpt_00254279680.pkl`

git `9aa46e7` · trained from `runs/g1-stairs-v11/run/ckpt_00190709760.pkl` · 300M steps · lr 0.0001 · scan `uniform`

**Certified envelope: stairs ≤ 6.4 cm** (0.7 m/s, 1.4 Hz cadence; required: strict, strict_camera) · target 10.0 cm: **NOT YET**

| condition | runs | crossed | fell | fall rate (95 % CI) | certified | torque-saturated | joint at limit |
|---|---|---|---|---|---|---|---|
| strict (required) | 96 | 91 | 5 | 5.2 % (2.2-11.6) | 6.4 cm | n/a | n/a |
| strict_camera (required) | 96 | 86 | 10 | 10.4 % (5.8-18.1) | 6.4 cm | n/a | n/a |
| speed_0.6 | 96 | 84 | 11 | 11.5 % (6.5-19.4) | 4.7 cm | 12.6 % | 13.0 % |
| tread_30cm | 32 | 30 | 2 | 6.2 % (1.7-20.1) | 12.0 cm | 13.1 % | 8.1 % |
| push | 32 | 28 | 4 | 12.5 % (5.0-28.1) | 9.9 cm | 13.2 % | 14.9 % |
| low_friction | 32 | 14 | 18 | 56.2 % (39.3-71.8) | 0.0 cm | 14.4 % | 14.5 % |
| payload_5kg | 32 | 28 | 4 | 12.5 % (5.0-28.1) | 9.9 cm | 9.7 % | 12.0 % |
| delay_20ms | 32 | 0 | 32 | 100.0 % (89.3-100.0) | 0.0 cm | 33.3 % | 48.2 % |

Per step height (crossed/runs, falls):

- **strict**: 3.0 cm 12/12 fell 0, 4.7 cm 12/12 fell 0, 6.4 cm 12/12 fell 0, 8.1 cm 11/12 fell 1, 9.9 cm 11/12 fell 1, 11.6 cm 12/12 fell 0, 13.3 cm 10/12 fell 2, 15.0 cm 11/12 fell 1
- **strict_camera**: 3.0 cm 12/12 fell 0, 4.7 cm 12/12 fell 0, 6.4 cm 12/12 fell 0, 8.1 cm 11/12 fell 1, 9.9 cm 8/12 fell 4, 11.6 cm 12/12 fell 0, 13.3 cm 9/12 fell 3, 15.0 cm 10/12 fell 2
- **speed_0.6**: 3.0 cm 12/12 fell 0, 4.7 cm 12/12 fell 0, 6.4 cm 11/12 fell 1, 8.1 cm 12/12 fell 0, 9.9 cm 11/12 fell 0, 11.6 cm 10/12 fell 2, 13.3 cm 9/12 fell 3, 15.0 cm 7/12 fell 5
- **tread_30cm**: 2.0 cm 4/4 fell 0, 4.0 cm 4/4 fell 0, 6.0 cm 4/4 fell 0, 8.0 cm 4/4 fell 0, 10.0 cm 4/4 fell 0, 12.0 cm 4/4 fell 0, 14.0 cm 3/4 fell 1, 16.0 cm 3/4 fell 1
- **push**: 3.0 cm 4/4 fell 0, 4.7 cm 4/4 fell 0, 6.4 cm 4/4 fell 0, 8.1 cm 4/4 fell 0, 9.9 cm 4/4 fell 0, 11.6 cm 3/4 fell 1, 13.3 cm 3/4 fell 1, 15.0 cm 2/4 fell 2
- **low_friction**: 3.0 cm 3/4 fell 1, 4.7 cm 4/4 fell 0, 6.4 cm 3/4 fell 1, 8.1 cm 2/4 fell 2, 9.9 cm 1/4 fell 3, 11.6 cm 0/4 fell 4, 13.3 cm 0/4 fell 4, 15.0 cm 1/4 fell 3
- **payload_5kg**: 3.0 cm 4/4 fell 0, 4.7 cm 4/4 fell 0, 6.4 cm 4/4 fell 0, 8.1 cm 4/4 fell 0, 9.9 cm 4/4 fell 0, 11.6 cm 3/4 fell 1, 13.3 cm 3/4 fell 1, 15.0 cm 2/4 fell 2
- **delay_20ms**: 3.0 cm 0/4 fell 4, 4.7 cm 0/4 fell 4, 6.4 cm 0/4 fell 4, 8.1 cm 0/4 fell 4, 9.9 cm 0/4 fell 4, 11.6 cm 0/4 fell 4, 13.3 cm 0/4 fell 4, 15.0 cm 0/4 fell 4
