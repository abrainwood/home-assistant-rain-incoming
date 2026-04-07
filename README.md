# Incoming Rain

A Home Assistant integration that detects whether rain is approaching your location using radar data, and predicts when it will arrive.

## Sensors

| Entity | Description |
|---|---|
| `binary_sensor.incoming_rain` | `on` when rain is moving toward your location within the configured lookahead window |
| `sensor.rain_arrival_time` | Predicted arrival timestamp, `unknown` when no rain incoming |

## How It Works

The integration fetches radar composite data from [RainViewer](https://www.rainviewer.com/) and analyses multiple frames to:

1. Filter out noise using spatial mass, temporal persistence, and directional coherence checks
2. Track rain cell motion using optical flow
3. Project rain cells forward and check whether they will reach your location within the lookahead window

## Configuration

| Option | Default | Range | Description |
|---|---|---|---|
| Latitude | HA location | - | Location to monitor |
| Longitude | HA location | - | Location to monitor |
| Lookahead | 60 min | 15-120 min | How far ahead to predict |

## Requirements

- Home Assistant 2024.1+
- Internet access (RainViewer API, free, no key required)

## Installation

Install via [HACS](https://hacs.xyz) or copy `custom_components/incoming_rain` into your HA config directory.
