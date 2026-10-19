# Backups to the author's Google Drive

Google Drive is the author's private storage and is authorised for backups
(`PLAN_v1.md` §1 rule 6). It is not publication. No rclone config, token or
credential appears in this repository, in any log, or in any commit.

## F0.1 — v0 checkpoint (phase F0, done before anything else)

- Remote: an rclone remote of type `drive`, named `gdrive` (name only; no config recorded).
- Remote folder: `gdrive:TiJepa/checkpoints_v0/`
- Command: `rclone copy checkpoints/ gdrive:TiJepa/checkpoints_v0/ --include "*.pt" --include "*.csv" --include "*.json"`
- Timestamp (UTC): 2026-10-15T23:41Z

| file | bytes | md5 (local) | md5 (remote) | match |
|---|---|---|---|---|
| `tijepa0_T1_seed0.pt` | 376538573 | `6abb61f5f80eacfbd0f92f1d671d4179` | `6abb61f5f80eacfbd0f92f1d671d4179` | yes |
| `pretrain_log_seed0.csv` | 132004 | — | `bb31c54acb53888adccc1a57d840096a` | yes (rclone check) |
| `pretrain_summary_seed0.json` | 997 | — | `fb09904e16948d7fa0c743f9889820d7` | yes (rclone check) |

Verification: `rclone check checkpoints/ gdrive:TiJepa/checkpoints_v0/ --one-way --include "*.pt"`
→ **0 differences found, 1 matching file**. The `.pt` md5 was additionally compared by
hand against `md5sum` of the local file and is identical.

**Status: O1 fixed.** The v0 weights now exist off the instance.

## F9.1 — v1 outputs

See the section appended at the end of the run.

## G0.5 — v1 partial checkpoint (step 14,751), on the v2 instance

- Timestamp (UTC): 2026-10-16T16:28Z
- **rclone could not run**: the `gdrive` remote exists on this instance but holds no OAuth
  token (rclone: "empty token found - please run rclone config reconnect"). Re-authorising
  needs the author's browser, so the planned copy to `gdrive:TiJepa/checkpoints_v1_partial/`
  is **pending**.
- What exists meanwhile:

| copy | location | bytes | hash | verified |
|---|---|---|---|---|
| original | `checkpoints/tijepa1_T1_seed0.pt` (this instance) | 376538701 | sha256 `2774e604…7d054b22`, md5 `86fa543bc2c0db40d7c5284da56e72e1` | G0.4 |
| frozen local copy | `checkpoints/v1_partial/tijepa1_T1_seed0.pt` | 376538701 | md5 `86fa543bc2c0db40d7c5284da56e72e1` | md5 equal |
| shared filesystem | `LambdaFilesystem/transfer/proj2/checkpoints/tijepa1_T1_seed0.pt` | 376538701 | sha256 `2774e604…7d054b22` | sha256 equal |
| Drive (uploaded by the previous instance at the move) | `gdrive:TiJepa/checkpoints_v1/tijepa1_T1_seed0.pt` | 376538701 | — | size and modification time (2026-10-16T04:25:59Z) match; md5 **not** verifiable without rclone |

The log CSV (`pretrain_log_tijepa1_seed0.csv`, md5 `6e951a290182b8a5355db55a9470c644`) is
in the local frozen copy and on the shared filesystem (md5 equal); it is not on Drive.
Resumed training writes only `checkpoints/tijepa1_T1_seed0.pt`; the frozen copies are never
written. The md5 check against Drive is re-run as soon as the remote is re-authorised.

## H0.2 — final checkpoints (v3 instance)

- Timestamp (UTC): 2026-10-19T01:00Z
- **Second location: Lambda persistent filesystem**, `LambdaFilesystem/transfer/proj2/tijepa_backups/`
  (PLAN_v3 §3.2 allows it as the second location until Drive works). Copied with `cp` + `sync`,
  then `sha256sum -c` run inside the backup folder against the hashes of the originals:
  **all 9 files OK**. The hash list is stored beside the files as `SHA256SUMS`.
- **Drive: still not possible.** The `gdrive` remote holds no OAuth token. The v2 backup loop
  (`98_backup_loop_v2.sh`) was stopped, because it could only log "remote unusable". It is
  re-run by hand once the author re-authorises rclone.

| file | bytes | sha256 (original = backup) |
|---|---|---|
| `tijepa0_T1_seed0.pt` | 376538573 | `f8ed6c89bf9a9df0f5d141b5c0a41c13d3295482acc27dba7851f1fbdd27cd92` |
| `tijepa1_T1_seed0.pt` (step 117,000, final) | 376537518 | `a09e76e91b5c52d884f484cfbf5b00951b162136e1a44f6995dd2829f0289d38` |
| `tijepa1_spatial_T1_seed0.pt` (step 117,000, final) | 376534574 | `e16cbafdf8ca86bf26c44715ff8aa978dfe20bb52bf96c788c8d5a1686546f13` |
| `pretrain_log_seed0.csv` | 132004 | `dd8158fe9b20b1032d587fc7a6530a8c3be1948b16b8a76524ab7893778727ec` |
| `pretrain_log_tijepa1_seed0.csv` | 411247 | `fb2992571f9f77868677d2ccf3753d584f8524516ac61de07f6047a5fcea7109` |
| `pretrain_log_tijepa1_spatial_seed0.csv` | 361197 | `694cf5df2ef6aa9c1b903ca03fcf671012630a0585edc3e2b6336257de3ceb20` |
| `pretrain_summary_seed0.json` | 2418 | `e772aae5c2cf4f4466dc48a1927a7b3fae0fb962e3dd601df0cd976691658f37` |
| `pretrain_summary_tijepa1_seed0.json` | 4663 | `3fc6022f9e464081749fb3aca3059e741fc4c141615568dbf237840127e513a7` |
| `pretrain_summary_tijepa1_spatial_seed0.json` | 3363 | `cf0d61edaaf4e8605f8b02a0148f75bfe767db4ba005e3ed9a2e1ba566d207da` |

TiJepa-0 also has its 2026-10-15 Drive copy (md5-verified, section above).

### Note on `pretrain_summary_seed0.json`

The copy on the persistent filesystem (sha256 `e772aae5…`) is the version overwritten by a
smoke run (see `reports/V0_ERRATA.md`, V2-E1); the original TiJepa-0 summary is only on
Drive (`gdrive:TiJepa/checkpoints_v0/`, md5 `fb09904e…`). The checkpoint and log are intact.

### v3 additions

| file | location | sha256 | verified |
|---|---|---|---|
| `trace_jepa_v3_seed0.pt` (trace JEPA, 6,000 steps) | persistent filesystem `tijepa_backups/` | `a77a9078606b3abe8fcc24e555aa9291140466cf5578682780b25900b3d7d1c0` | yes |

Sweep snapshots are added when the sweep finishes.
