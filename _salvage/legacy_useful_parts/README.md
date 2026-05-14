# Legacy Useful Parts

This directory is a holding area for useful code or ideas salvaged from the old GUI/API/chat/session shell.

It is not active Harness Native Reserving Studio architecture.

Rules:

- Do not add new runtime dependencies from harness tools into this directory.
- Do not make new harness tools depend on old GUI/API/chat/session/control-plane modules.
- If a helper becomes useful, promote it into `source/` or `harness/` first and test it there.
- Keep deterministic reserving core and domain services active in their current locations.
