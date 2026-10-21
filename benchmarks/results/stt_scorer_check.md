# Scorer sanity checks

## 1. Reference text vs itself (must be WER 0 and numbers match)

| id | extracted numbers from the reference | self WER |
|---|---|---|
| num01 | ['55000'] | 0.0 |
| num02 | ['87500'] | 0.0 |
| num03 | ['48000'] | 0.0 |
| num04 | ['120000'] | 0.0 |
| num05 | ['162900'] | 0.0 |
| num06 | ['35000', '60000'] | 0.0 |
| num07 | ['72500'] | 0.0 |
| num08 | ['2019', '40000'] | 0.0 |
| plate01 | ['12345'] | 0.0 |
| plate02 | ['34567'] | 0.0 |
| plate03 | ['30'] | 0.0 |
| phone01 | ['0501234567'] | 0.0 |
| phone02 | ['0559876543'] | 0.0 |
| phone03 | ['971504443322'] | 0.0 |
| date01 | ['34'] | 0.0 |
| date02 | ['10'] | 0.0 |
| date03 | ['20', '2'] | 0.0 |

## 2. Correct digit-style renderings (Whisper style) must compare equal to the reference

| id | digit rendering | ref numbers | hyp numbers | equal |
|---|---|---|---|---|
| num01 | My budget is 55,000 dirhams. | ['55000'] | ['55000'] | yes |
| num04 | I can pay 120,000. | ['120000'] | ['120000'] | yes |
| num05 | The price is 162,900. | ['162900'] | ['162900'] | yes |
| num06 | Under 35,000 and less than 60,000 kilometres. | ['35000', '60000'] | ['35000', '60000'] | yes |
| num08 | A 2019 model with 40,000 kilometres. | ['2019', '40000'] | ['2019', '40000'] | yes |
| car02 | Do you have a Nissan Patrol from 2021? | ['2021'] | ['2021'] | yes |
| phone01 | My number is 050 123 4567. | ['0501234567'] | ['0501234567'] | yes |
| phone03 | +971 50 444 3322 | ['971504443322'] | ['971504443322'] | yes |
| plate01 | The plate is Dubai A 12345. | ['12345'] | ['12345'] | yes |
| plate02 | Plate number Sharjah 34567. | ['34567'] | ['34567'] | yes |
| date01 | Can I come on Saturday at 4:30 in the afternoon? | ['34'] | ['34'] | yes |
| date03 | Let us do the 25th of November at 2 p.m. | ['20', '2'] | ['25', '2'] | **NO (scorer issue)** |
