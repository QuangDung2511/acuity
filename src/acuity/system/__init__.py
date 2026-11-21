# Platform adapters: paths, camera backend, power, foreground window, tray.
# ─────────────────────────────────────────────────────────────────
# Bottom layer (with db, utils): never imports core or ui.
# Each adapter with OS-specific code is chosen by a create_*() factory on
# sys.platform (paths has none: Qt covers every OS), and only
# the factory imports an OS-specific module, so platform extras stay optional.
# ─────────────────────────────────────────────────────────────────
