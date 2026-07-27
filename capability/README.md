# Historical campaign-specific evaluations

This directory currently contains the frozen Record Patch prototype. Its task,
data, evaluator, splits, thresholds, and protocols were choices inside that
experiment. They are not requirements imposed on new OrcaColony campaigns.

New campaign owners define their own usage scenario and evaluation contract
through
[`../research/CAMPAIGN_FRAMEWORK.md`](../research/CAMPAIGN_FRAMEWORK.md).
Campaign-specific evaluator files may live here, in the campaign's data
repository, or in another durable location chosen by the owner, provided the
campaign pins their exact revisions and the release preserves the evidence.

The retained prototype is [`record-patch-v1`](record-patch-v1/TASK.md).
