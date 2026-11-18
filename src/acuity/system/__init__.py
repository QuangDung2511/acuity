# Platform adapters: paths, camera backend, power, foreground window, tray.
# ─────────────────────────────────────────────────────────────────
# Bottom layer (with db, utils): never imports core or ui.
# Each adapter is chosen by a create_*() factory on sys.platform, and only
# the factory imports an OS-specific module, so platform extras stay optional.
# ─────────────────────────────────────────────────────────────────
