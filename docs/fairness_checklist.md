# Fairness Checklist

Use this checklist before claiming that a comparison is fair.

## Data and Task Controls

- [ ] Same dataset and task definition for all compared models
- [ ] Same data split for all compared models
- [ ] Same label space
- [ ] Same concept supervision access where relevant
- [ ] Same preprocessing pipeline where possible

## Training Controls

- [ ] Same random seed policy
- [ ] Same epoch budget or equivalent stopping rule
- [ ] Same optimizer family unless deviation is justified
- [ ] Comparable batch size where feasible
- [ ] Comparable early stopping policy if used

## Capacity and Cost Controls

- [ ] Trainable parameter count recorded
- [ ] Runtime recorded
- [ ] Checkpointing policy recorded
- [ ] Hardware context recorded
- [ ] Training budget documented

## Reporting Controls

- [ ] Final task metrics reported
- [ ] Concept-level metrics reported where applicable
- [ ] Semantic consistency reported where applicable
- [ ] Standard and controlled settings clearly separated
- [ ] Deviations from perfect control explicitly stated

## TODO

- TODO: Add exact threshold rules for what counts as "similar parameter budget"
- TODO: Add the final seed list used in experiments
- TODO: Add a concise hardware reporting template before running full experiments
