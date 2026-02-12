# Cross-Tab Sync (Per User, Local Browser)

This project now syncs dashboard updates across open tabs without 1-second polling.

## What was changed

- Removed periodic polling (`dcc.Interval`) for sync.
- Added browser event-based sync using a small JS bridge in `assets/tab_sync.js`.
- Added `sync_version` to session persistence to detect stale updates safely.
- Dash is explicitly configured to load assets from the repository `assets/` directory.

## How it works

1. A tab changes a reserving setting (for example, drops, tail, or BF apriori).
2. Python recalculates and saves the session YAML.
3. During save, `sync_version` is incremented by 1.
4. The tab publishes a lightweight event to other tabs:
   - `type`: `session_changed`
   - `user_key`: current segment (for example `quarterly`)
   - `sync_version`: incremented integer
   - `origin_tab_id`: sender tab id
   - `updated_at`: ISO timestamp
5. Other tabs receive the event, ignore self-events, and compare versions.
6. If incoming version is newer, they reload session settings and recalculate.

## Why JavaScript is used

Browser APIs (`BroadcastChannel`, `storage` events) are only available in JavaScript.
Python remains the source of truth for model calculations and session persistence.

## Loop and stale update protection

- Self-events are ignored (`origin_tab_id` check).
- Older or equal versions are ignored (`incoming <= current`).
- Sync is scoped by `user_key` so unrelated contexts do not cross-talk.

## Fallback behavior

- Primary transport: `BroadcastChannel`.
- Fallback transport: `localStorage` + `storage` event.
- The storage listener is initialized on load to avoid startup race conditions between tabs.

## Dash bridge detail

- Incoming sync events are written into hidden Dash input `sync-inbox`.
- The JS bridge uses the native input value setter and dispatches both `input` and `change` events so Dash/React reliably detects updates.

## Future roadmap (lightweight progression)

1. Extract sync/session callback logic from `Dashboard` into a focused helper module.
2. Add tests for version handling and stale event rejection.
3. Keep the same event contract when moving to optional database/server push later.
