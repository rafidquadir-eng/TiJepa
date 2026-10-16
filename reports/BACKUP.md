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
