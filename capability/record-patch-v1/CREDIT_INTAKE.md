# Record Patch contributor credit intake

Training contributors use `orcacolony_participants_v2`. Before work is accepted,
the campaign operator records:

- a private stable contributor ID;
- one or more private worker IDs and token hashes;
- named, pseudonymous, or anonymous public credit;
- optional display name, HTTPS profile, team, and role labels;
- whether accepted assignment and token totals may be shown;
- whether a contributor-supplied hardware class may be shown.

The coordinator records accepted assignments and tokens. Worker runtime and
hardware descriptions are contributor-reported and are labeled that way.
Release generation freezes the current choices into
`attribution-snapshot.json` and `CONTRIBUTORS.md`.

Use
[`../../campaign/record-patch-t2-participants.example.json`](../../campaign/record-patch-t2-participants.example.json)
as the campaign-specific intake shape. Store the real manifest and raw worker
tokens outside Git. Only token hashes belong in the manifest.

Data design, evaluator implementation, report review, campaign operation,
hosting, and useful failed compute attempts are not automatically counted by the
accepted-training ledger. Record those people and their chosen public identity
in the capability report with a link to the work they contributed. Do not label
their time or hardware as coordinator-verified. A machine-readable auxiliary
contribution ledger remains required before those totals can be aggregated
automatically.
