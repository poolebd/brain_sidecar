# Meeting Cockpit Screenshots

These PNGs are intentionally committed UI review artifacts for the dark-mode meeting cockpit. They are generated with mocked local data and are not runtime product assets.

| Page | Screenshot |
| --- | --- |
| Live | ![Live page](live.png) |
| Sessions | ![Sessions page](sessions.png) |
| Tools | ![Tools page](tools.png) |
| Tools - Voice & Input | ![Tools Voice & Input panel](tools-voice-input.png) |
| Tools - Speaker Identity | ![Tools Speaker Identity panel](tools-speaker-identity.png) |
| Tools - Test Mode | ![Tools Test Mode panel](tools-test-mode.png) |
| Tools - Logs & Debug | ![Tools Logs & Debug panel](tools-logs-debug.png) |
| Tools - Appearance | ![Tools Appearance panel](tools-appearance.png) |
| Models | ![Models page](models.png) |
| Memory | ![Memory page](memory.png) |

Captured at `1440 x 900` with Playwright.

Regenerate explicitly from the repo root:

```bash
BRAIN_SIDECAR_CAPTURE_SCREENSHOTS=1 npm --prefix ui run test:e2e -- ui/e2e/screenshot-pages.spec.ts
```

Normal e2e runs skip screenshot regeneration so these files do not churn accidentally.
