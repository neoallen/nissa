# Watch Timer

Mechanical watch timing analysis tool. Reads watch tick audio and computes daily rate deviation (s/d), beat error (ms), and amplitude (&#176;).

Python port of the [tg](https://github.com/vacaboja/tg) timing software by Marcello Mamino.

## Features

- **CLI mode:** analyze `.wav`/`.m4a` recordings from command line
- **Web mode:** record directly from iPhone/Android microphone via ngrok tunnel
- **95% confidence intervals** for rate estimates via bootstrap
- IQR-based outlier rejection for robust multi-chunk aggregation

## Quick Start

```bash
# Install dependencies
uv sync

# CLI — analyze an audio file
uv run tg-python/tg_timer.py recording.m4a [bph] [lift_angle]

# Web — start server with ngrok tunnel (iPhone access)
start.bat
# or
uv run web_app.py
```

## CLI Usage

```
uv run tg-python/tg_timer.py <audio_file> [bph] [la] [--json]
```

| Arg | Default | Description |
|-----|---------|-------------|
| `audio_file` | (required) | Path to `.wav` or `.m4a` recording |
| `bph` | 21600 | Beats per hour (or `0` for auto-detect) |
| `la` | 52.0 | Lift angle in degrees |
| `--json` | — | Output machine-readable JSON |

### Examples

```bash
uv run tg-python/tg_timer.py watch_recording.m4a
uv run tg-python/tg_timer.py vostok.m4a 18000
uv run tg-python/tg_timer.py recording.m4a --json > result.json
```

## Web App

```
start.bat
```

1. ngrok public URL is displayed as a QR code in the terminal
2. Scan the QR code with your iPhone camera, or open the URL in Safari
3. Select BPH and lift angle, tap **REC** to start recording
4. Place your phone near the watch movement
5. Tap again to stop — results appear automatically

### Without ngrok

```bash
uv run web_app.py --no-ngrok
# Open http://localhost:8000 in your browser
```

## Requirements

- Python 3.14+
- [ffmpeg](https://ffmpeg.org/) (for non-WAV audio decoding)
- [ngrok](https://ngrok.com/) (for web app remote access; install via `winget install Ngrok.Ngrok`)
- ngrok auth token (set in `.env` as `NGROK_AUTH_TOKEN=your_token`)

## Output

| Metric | Description |
|--------|-------------|
| Rate | Daily deviation in seconds per day (positive = fast) |
| 95% CI | Bootstrap confidence interval for the rate |
| Period | Tick interval in milliseconds |
| BPH | Measured beats per hour |
| Beat Error | Impulse asymmetry in milliseconds |
| Amplitude | Balance wheel swing angle in degrees |

## License

GPL-3.0 (matches original [tg](https://github.com/vacaboja/tg) license)
