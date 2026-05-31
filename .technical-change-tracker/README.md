# Technical Change Tracker

This folder stores structured technical change records for AI/session handoff.

State flow:

```text
planned -> in_progress -> implemented -> tested -> deployed
             |
             +-> blocked
```

Files:

- `state.json`: tracker metadata and active change IDs.
- `changes/`: append-only JSON change records.
- `dashboard.html`: lightweight local status view.

Recommended commands by convention:

- `/tc create`: add a new JSON record in `changes/`.
- `/tc update`: append status/context/test evidence to a record.
- `/tc status`: summarize active records.
- `/tc close`: mark a record deployed or blocked.
