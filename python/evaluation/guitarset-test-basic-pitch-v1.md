# Transcriber Quality Evaluation

Corpus: **GuitarSet v1.1.0**; backend `onnx` / mode `full`; 60 clips.

Build: `GNU 16.1.0` / `Release`; model `onnx:2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec` / SHA-256 `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`; provider `cpu`.

Timing protocol: 1 measured repetitions, 0 warmups; new process per clip and repetition; model/session cold.

## Overall metrics

| Metric | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| note_event | 0.0921 | 0.1874 | 0.1235 | 1633 | 16105 | 7082 |
| onset_pitch | 0.2308 | 0.4698 | 0.3095 | 4094 | 13644 | 4621 |
| pitch_event | 0.4331 | 0.8815 | 0.5808 | 7682 | 10056 | 1033 |
| frame | 0.7512 | 0.7023 | 0.7260 | 192803 | 63857 | 81709 |

False positives/min: **528.589**; fragmented reference notes: **4751**; merged predictions: **1498**.

## Dataset provenance

Corpus identity: `bb2616779be01c203d0b6d276cbbe6b55218928cf96c462189e4759acb1e0bc9`.

```json
{
  "dataset_split_policy": {
    "grouping": "performer ID; all recordings by a performer stay in one partition",
    "partitions": {
      "test": [
        "05"
      ],
      "train": [
        "00",
        "01",
        "02",
        "03"
      ],
      "validation": [
        "04"
      ]
    },
    "player_assignments": {
      "00": "train",
      "01": "train",
      "02": "train",
      "03": "train",
      "04": "validation",
      "05": "test"
    },
    "warning": "The grouped split prevents performer overlap within this protocol, but the Basic Pitch model may have seen GuitarSet recordings during training."
  },
  "provenance": {
    "archive_md5": {
      "annotations": "b39b78e63d3446f2e54ddb7a54df9b10",
      "audio": "275966d6610ac34999b58426beb119c3"
    },
    "attribution": "Qingyang Xi et al., ISMIR 2018",
    "doi": "10.5281/zenodo.3371780",
    "license": "CC BY 4.0",
    "possible_model_training_overlap": true
  },
  "selected_split": "test"
}
```

### onnx / full
Model `onnx:2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`; SHA-256 `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`; provider `cpu`.

| Metric | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| note_event | 0.0921 | 0.1874 | 0.1235 | 1633 | 16105 | 7082 |
| onset_pitch | 0.2308 | 0.4698 | 0.3095 | 4094 | 13644 | 4621 |
| pitch_event | 0.4331 | 0.8815 | 0.5808 | 7682 | 10056 | 1033 |
| frame | 0.7512 | 0.7023 | 0.7260 | 192803 | 63857 | 81709 |

#### Requested benchmark metrics

| Metric | Score (0-1) | Definition |
|---|---:|---|
| Note precision | 0.0921 | Exact-pitch note events matched on onset and offset |
| Note recall | 0.1874 | Exact-pitch note events matched on onset and offset |
| F1 score | 0.1235 | Micro-aggregated strict note-event F1 |
| Onset accuracy | 0.4698 | Exact-pitch onset recall within the configured onset tolerance |
| Offset accuracy | 0.3920 | Offset-tolerance pass rate among exact-pitch onset matches |
| Polyphonic accuracy | 0.7260 | Framewise pitch-activity F1 at 0.01 s |

False positives/min: **528.589**; fragmented references: **4751** (rate 0.545); merged predictions: **1498** (rate 0.084).

| Timing/resource | Median | p95 | Samples |
|---|---:|---:|---:|
| analysis_seconds | 0.7869 s | 1.1464 s | 60 |
| total_seconds | 0.7894 s | 1.1502 s | 60 |
| process_wall_seconds | 0.8224 s | 1.1832 s | 60 |
| peak_resident_bytes | 79263744.0000 bytes | 89210880.0000 bytes | 60 |

Engine corpus real-time factor: 0.02653.

#### Scenario cohorts

| Scenario | Clips | Event F1 | Pitch F1 | Frame F1 | FP/min | Fragmented refs | Merged predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| comp | 30 | 0.1163 | 0.5650 | 0.7338 | 800.318 | 3520 | 1377 |
| guitarset | 60 | 0.1235 | 0.5808 | 0.7260 | 528.589 | 4751 | 1498 |
| solo | 30 | 0.1452 | 0.6287 | 0.7024 | 256.861 | 1231 | 121 |
| test | 60 | 0.1235 | 0.5808 | 0.7260 | 528.589 | 4751 | 1498 |

#### Register cohorts

| Register (MIDI) | Event F1 | Pitch F1 | Frame F1 | Reference notes |
|---|---:|---:|---:|---:|
| high_midi_72_127 | 0.1641 | 0.6615 | 0.6865 | 426 |
| low_midi_0_47 | 0.0611 | 0.3896 | 0.7359 | 687 |
| mid_midi_48_71 | 0.1302 | 0.6037 | 0.7265 | 7602 |

#### Per-clip results (regressions remain visible)

| Clip | Scenarios | Refs | Preds | Event F1 | Onset+pitch F1 | Pitch F1 | Frame F1 | Onset median (s) | Offset median (s) | FP/min | Fragmented refs | Merged preds |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 05_BN1-129-Eb_comp | guitarset, comp, test | 148 | 420 | 0.0211 | 0.3380 | 0.5106 | 0.8071 | 0.0259 | 0.1951 | 1112.684 | 123 | 6 |
| 05_BN1-129-Eb_solo | guitarset, solo, test | 44 | 104 | 0.1486 | 0.4189 | 0.5946 | 0.8426 | 0.0251 | 0.0725 | 249.951 | 38 | 0 |
| 05_BN1-147-Gb_comp | guitarset, comp, test | 96 | 359 | 0.0396 | 0.3209 | 0.4088 | 0.8518 | 0.0212 | 0.2269 | 1072.216 | 82 | 6 |
| 05_BN1-147-Gb_solo | guitarset, solo, test | 27 | 83 | 0.1818 | 0.3636 | 0.4545 | 0.7432 | 0.0228 | 0.0679 | 223.634 | 16 | 2 |
| 05_BN2-131-B_comp | guitarset, comp, test | 202 | 532 | 0.0272 | 0.2807 | 0.5232 | 0.8067 | 0.0214 | 0.1953 | 1068.709 | 146 | 6 |
| 05_BN2-131-B_solo | guitarset, solo, test | 69 | 167 | 0.1525 | 0.2966 | 0.5678 | 0.6280 | 0.0233 | 0.0512 | 305.053 | 38 | 4 |
| 05_BN2-166-Ab_comp | guitarset, comp, test | 173 | 432 | 0.0364 | 0.3140 | 0.5289 | 0.7791 | 0.0212 | 0.1506 | 1092.088 | 118 | 3 |
| 05_BN2-166-Ab_solo | guitarset, solo, test | 67 | 126 | 0.2798 | 0.3834 | 0.6839 | 0.7414 | 0.0211 | 0.0443 | 256.809 | 45 | 3 |
| 05_BN3-119-G_comp | guitarset, comp, test | 180 | 585 | 0.0078 | 0.2431 | 0.4575 | 0.8071 | 0.0257 | 0.2102 | 1082.348 | 150 | 3 |
| 05_BN3-119-G_solo | guitarset, solo, test | 62 | 155 | 0.1014 | 0.2396 | 0.5346 | 0.6785 | 0.0324 | 0.0961 | 267.797 | 40 | 0 |
| 05_BN3-154-E_comp | guitarset, comp, test | 185 | 408 | 0.0607 | 0.3204 | 0.5599 | 0.7775 | 0.0243 | 0.1439 | 938.553 | 124 | 7 |
| 05_BN3-154-E_solo | guitarset, solo, test | 56 | 117 | 0.1618 | 0.3121 | 0.6474 | 0.7193 | 0.0249 | 0.0556 | 247.874 | 37 | 0 |
| 05_Funk1-114-Ab_comp | guitarset, comp, test | 181 | 224 | 0.2864 | 0.4840 | 0.7259 | 0.6698 | 0.0300 | 0.0389 | 394.324 | 58 | 43 |
| 05_Funk1-114-Ab_solo | guitarset, solo, test | 60 | 105 | 0.1939 | 0.3758 | 0.6424 | 0.6325 | 0.0255 | 0.0510 | 211.415 | 27 | 3 |
| 05_Funk1-97-C_comp | guitarset, comp, test | 177 | 182 | 0.2507 | 0.4067 | 0.8524 | 0.7606 | 0.0256 | 0.0434 | 276.876 | 22 | 22 |
| 05_Funk1-97-C_solo | guitarset, solo, test | 58 | 92 | 0.1067 | 0.2133 | 0.6533 | 0.7104 | 0.0220 | 0.0616 | 169.763 | 21 | 2 |
| 05_Funk2-108-Eb_comp | guitarset, comp, test | 245 | 338 | 0.1544 | 0.3431 | 0.6758 | 0.6868 | 0.0300 | 0.0688 | 494.537 | 86 | 55 |
| 05_Funk2-108-Eb_solo | guitarset, solo, test | 61 | 122 | 0.1530 | 0.2186 | 0.6448 | 0.8185 | 0.0171 | 0.0443 | 182.287 | 38 | 0 |
| 05_Funk2-119-G_comp | guitarset, comp, test | 182 | 273 | 0.2549 | 0.4088 | 0.7121 | 0.7009 | 0.0266 | 0.0370 | 399.836 | 72 | 40 |
| 05_Funk2-119-G_solo | guitarset, solo, test | 99 | 189 | 0.1181 | 0.2222 | 0.6597 | 0.6168 | 0.0209 | 0.0517 | 319.869 | 46 | 1 |
| 05_Funk3-112-C#_comp | guitarset, comp, test | 212 | 209 | 0.2565 | 0.4371 | 0.7363 | 0.6687 | 0.0262 | 0.0454 | 271.260 | 45 | 52 |
| 05_Funk3-112-C#_solo | guitarset, solo, test | 70 | 153 | 0.1525 | 0.2422 | 0.6099 | 0.7922 | 0.0252 | 0.0456 | 238.009 | 53 | 10 |
| 05_Funk3-98-A_comp | guitarset, comp, test | 184 | 352 | 0.0821 | 0.2687 | 0.5784 | 0.6765 | 0.0261 | 0.0914 | 505.375 | 94 | 45 |
| 05_Funk3-98-A_solo | guitarset, solo, test | 65 | 139 | 0.0980 | 0.2059 | 0.5784 | 0.7409 | 0.0188 | 0.0536 | 197.556 | 35 | 1 |
| 05_Jazz1-130-D_comp | guitarset, comp, test | 141 | 208 | 0.1547 | 0.3954 | 0.7335 | 0.7862 | 0.0239 | 0.0767 | 490.355 | 68 | 11 |
| 05_Jazz1-130-D_solo | guitarset, solo, test | 58 | 136 | 0.2165 | 0.3505 | 0.5979 | 0.6804 | 0.0243 | 0.0493 | 311.552 | 39 | 3 |
| 05_Jazz1-200-B_comp | guitarset, comp, test | 175 | 137 | 0.3141 | 0.5513 | 0.6859 | 0.5759 | 0.0288 | 0.0464 | 366.667 | 14 | 5 |
| 05_Jazz1-200-B_solo | guitarset, solo, test | 50 | 83 | 0.3459 | 0.5865 | 0.7519 | 0.7934 | 0.0216 | 0.0498 | 250.000 | 27 | 2 |
| 05_Jazz2-110-Bb_comp | guitarset, comp, test | 252 | 380 | 0.0633 | 0.2310 | 0.6804 | 0.6805 | 0.0283 | 0.1096 | 618.854 | 95 | 12 |
| 05_Jazz2-110-Bb_solo | guitarset, solo, test | 103 | 198 | 0.1528 | 0.2857 | 0.6645 | 0.7470 | 0.0221 | 0.0459 | 300.832 | 51 | 4 |
| 05_Jazz2-187-F#_comp | guitarset, comp, test | 247 | 196 | 0.1941 | 0.3792 | 0.6817 | 0.5526 | 0.0266 | 0.0479 | 447.107 | 22 | 10 |
| 05_Jazz2-187-F#_solo | guitarset, solo, test | 61 | 113 | 0.1724 | 0.4483 | 0.6897 | 0.7394 | 0.0213 | 0.0560 | 286.382 | 41 | 5 |
| 05_Jazz3-137-Eb_comp | guitarset, comp, test | 235 | 338 | 0.1396 | 0.3002 | 0.7120 | 0.6536 | 0.0316 | 0.0602 | 638.043 | 58 | 4 |
| 05_Jazz3-137-Eb_solo | guitarset, solo, test | 77 | 160 | 0.1266 | 0.2954 | 0.6498 | 0.7205 | 0.0212 | 0.0639 | 310.457 | 55 | 5 |
| 05_Jazz3-150-C_comp | guitarset, comp, test | 227 | 241 | 0.1624 | 0.3632 | 0.7265 | 0.6331 | 0.0256 | 0.0651 | 475.781 | 30 | 13 |
| 05_Jazz3-150-C_solo | guitarset, solo, test | 69 | 137 | 0.1748 | 0.3398 | 0.6408 | 0.7436 | 0.0297 | 0.0609 | 278.906 | 45 | 2 |
| 05_Rock1-130-A_comp | guitarset, comp, test | 312 | 419 | 0.3311 | 0.6211 | 0.7798 | 0.8139 | 0.0330 | 0.0466 | 807.325 | 210 | 179 |
| 05_Rock1-130-A_solo | guitarset, solo, test | 60 | 115 | 0.2286 | 0.4457 | 0.6743 | 0.7974 | 0.0298 | 0.0618 | 257.369 | 45 | 7 |
| 05_Rock1-90-C#_comp | guitarset, comp, test | 277 | 494 | 0.0960 | 0.3709 | 0.6667 | 0.7888 | 0.0331 | 0.1746 | 856.875 | 208 | 150 |
| 05_Rock1-90-C#_solo | guitarset, solo, test | 99 | 170 | 0.1487 | 0.2825 | 0.7063 | 0.6085 | 0.0319 | 0.0551 | 281.250 | 39 | 4 |
| 05_Rock2-142-D_comp | guitarset, comp, test | 460 | 414 | 0.3249 | 0.5080 | 0.7483 | 0.6717 | 0.0317 | 0.0321 | 603.514 | 189 | 178 |
| 05_Rock2-142-D_solo | guitarset, solo, test | 86 | 161 | 0.1377 | 0.3401 | 0.6721 | 0.6787 | 0.0254 | 0.0522 | 319.508 | 50 | 8 |
| 05_Rock2-85-F_comp | guitarset, comp, test | 325 | 550 | 0.0891 | 0.3109 | 0.6034 | 0.6610 | 0.0345 | 0.1910 | 678.721 | 181 | 107 |
| 05_Rock2-85-F_solo | guitarset, solo, test | 125 | 209 | 0.1317 | 0.2515 | 0.6587 | 0.5307 | 0.0288 | 0.0513 | 248.377 | 45 | 11 |
| 05_Rock3-117-Bb_comp | guitarset, comp, test | 398 | 483 | 0.3133 | 0.5108 | 0.7560 | 0.7191 | 0.0370 | 0.0385 | 630.782 | 222 | 176 |
| 05_Rock3-117-Bb_solo | guitarset, solo, test | 70 | 148 | 0.1468 | 0.2752 | 0.6239 | 0.8138 | 0.0264 | 0.0746 | 241.343 | 55 | 9 |
| 05_Rock3-148-C_comp | guitarset, comp, test | 249 | 474 | 0.1743 | 0.2490 | 0.6584 | 0.7456 | 0.0413 | 0.0386 | 950.626 | 206 | 196 |
| 05_Rock3-148-C_solo | guitarset, solo, test | 61 | 126 | 0.1070 | 0.2674 | 0.6310 | 0.7834 | 0.0261 | 0.0625 | 268.303 | 41 | 9 |
| 05_SS1-100-C#_comp | guitarset, comp, test | 102 | 601 | 0.0057 | 0.1650 | 0.2845 | 0.8167 | 0.0213 | 0.4331 | 1247.917 | 96 | 7 |
| 05_SS1-100-C#_solo | guitarset, solo, test | 60 | 132 | 0.1354 | 0.2708 | 0.6042 | 0.6460 | 0.0248 | 0.0596 | 247.917 | 34 | 5 |
| 05_SS1-68-E_comp | guitarset, comp, test | 143 | 798 | 0.0170 | 0.1743 | 0.3039 | 0.7754 | 0.0245 | 0.2685 | 1119.340 | 133 | 10 |
| 05_SS1-68-E_solo | guitarset, solo, test | 78 | 186 | 0.0606 | 0.1515 | 0.5000 | 0.5994 | 0.0177 | 0.0799 | 252.206 | 32 | 2 |
| 05_SS2-107-Ab_comp | guitarset, comp, test | 222 | 642 | 0.0208 | 0.2315 | 0.4722 | 0.7222 | 0.0214 | 0.2313 | 1058.340 | 163 | 2 |
| 05_SS2-107-Ab_solo | guitarset, solo, test | 87 | 167 | 0.1339 | 0.2677 | 0.6614 | 0.7200 | 0.0228 | 0.0530 | 250.791 | 43 | 3 |
| 05_SS2-88-F_comp | guitarset, comp, test | 196 | 992 | 0.0118 | 0.1684 | 0.3199 | 0.7478 | 0.0225 | 0.5168 | 1354.597 | 166 | 11 |
| 05_SS2-88-F_solo | guitarset, solo, test | 93 | 180 | 0.0806 | 0.2344 | 0.6154 | 0.6955 | 0.0253 | 0.0618 | 232.413 | 46 | 5 |
| 05_SS3-84-Bb_comp | guitarset, comp, test | 211 | 839 | 0.0343 | 0.1733 | 0.4019 | 0.7021 | 0.0253 | 0.1905 | 1077.720 | 177 | 3 |
| 05_SS3-84-Bb_solo | guitarset, solo, test | 111 | 225 | 0.1071 | 0.2083 | 0.6131 | 0.6309 | 0.0248 | 0.0501 | 271.727 | 56 | 4 |
| 05_SS3-98-C_comp | guitarset, comp, test | 199 | 828 | 0.0312 | 0.2162 | 0.3700 | 0.7523 | 0.0205 | 0.2939 | 1243.530 | 162 | 15 |
| 05_SS3-98-C_solo | guitarset, solo, test | 93 | 192 | 0.1193 | 0.2246 | 0.6035 | 0.6689 | 0.0251 | 0.0612 | 268.002 | 53 | 7 |
