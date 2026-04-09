# Incoming Rain

A Home Assistant integration that detects whether rain is approaching your location using radar data, and predicts when it will arrive.

## Sensors

| Entity | Type | Description |
|---|---|---|
| `binary_sensor.incoming_rain_status` | Binary | `on` when rain is approaching or overhead |
| `sensor.incoming_rain_arrival_time` | Timestamp | Predicted arrival time, `unknown` when no rain incoming |
| `sensor.incoming_rain_intensity` | Sensor | Precipitation intensity (none/light/moderate/heavy/extreme) |
| `sensor.incoming_rain_last_rain` | Timestamp | Last time rain was detected nearby |
| `image.incoming_rain_radar_64km` | Image | Radar composite map - 64km radius |
| `image.incoming_rain_radar_128km` | Image | Radar composite map - 128km radius |
| `image.incoming_rain_radar_256km` | Image | Radar composite map - 256km radius |
| `image.incoming_rain_radar_512km` | Image | Radar composite map - 512km radius |

The binary sensor and arrival time sensor expose these attributes for debugging and automation:

| Attribute | Description |
|---|---|
| `latitude` | Monitored location latitude |
| `longitude` | Monitored location longitude |
| `lookahead_minutes` | How far ahead the integration is looking |
| `confidence` | `normal` (3+ frames), `degraded` (2 frames), or `unavailable` |
| `frame_count` | Number of radar frames used in the analysis |

## How it works

The integration fetches radar composite data from [RainViewer](https://www.rainviewer.com/) and analyses multiple frames to:

1. Filter out noise using spatial mass, temporal persistence, and directional coherence checks
2. Track rain cell motion by matching centroids across frames
3. Project rain cells forward and check whether they will reach your location within the lookahead window
4. Report rain as incoming if it's overhead OR projected to arrive within the window

## Installation

Install via [HACS](https://hacs.xyz) or copy `custom_components/incoming_rain` into your HA config directory.

## Configuration

This integration is configured entirely through the Home Assistant UI - no YAML needed.

**Settings -> Integrations -> Add Integration -> "Incoming Rain"**

| Option | Default | Range | Description |
|---|---|---|---|
| Latitude | Your HA home location | -90 to 90 | Location to monitor |
| Longitude | Your HA home location | -180 to 180 | Location to monitor |
| Lookahead | 60 min | 15-120 min | How far ahead to predict |

The latitude and longitude default to your Home Assistant home location (set in Settings -> General).

### Changing settings after setup

To change the location or lookahead after initial setup:

**Settings -> Integrations -> Incoming Rain -> Configure**

Changes take effect immediately - the coordinator will refetch radar data on the next poll.

## Use cases

- Close a pergola cover when rain is detected approaching
- Recall a robotic lawnmower before it gets wet
- Send a notification when rain will arrive within 30 minutes
- Trigger automations based on confidence level (normal vs degraded)

## Example automations

### Close pergola cover when rain is incoming

```yaml
automation:
  - alias: "Close pergola cover - rain incoming"
    trigger:
      - platform: state
        entity_id: binary_sensor.incoming_rain_status
        to: "on"
    action:
      - service: cover.close_cover
        target:
          entity_id: cover.pergola
```

### Return lawnmower before rain arrives

```yaml
automation:
  - alias: "Return lawnmower - rain incoming"
    trigger:
      - platform: state
        entity_id: binary_sensor.incoming_rain_status
        to: "on"
    condition:
      - condition: not
        conditions:
          - condition: state
            entity_id: sensor.incoming_rain_intensity
            state: "light"
    action:
      - service: vacuum.return_to_base
        target:
          entity_id: vacuum.lawnmower
```

## Troubleshooting

### Confidence levels

- **normal** - 3 or more radar frames available. Detection is running at full accuracy.
- **degraded** - only 2 frames available. Detection still works but motion tracking is less reliable. This can happen when the RainViewer API is temporarily returning fewer frames.
- **unavailable** - fewer than 2 frames. The sensors will show as unavailable until more data arrives.

### Cold-start period

The integration learns your local clutter patterns over 12+ hours. During this period you may see slightly reduced accuracy.

### False positives

Anomalous propagation (AP) - radar artifacts that appear during clear, still nights. The QC system progressively filters these as it learns.

### No rain detected when it's clearly raining

Check that your configured latitude and longitude are correct. The crosshair on the radar image entities shows the monitored location - verify it matches where you expect.

## Requirements

- Home Assistant 2024.1+
- Internet access (RainViewer API - free, no API key required)

## Data source

This integration uses the [RainViewer API](https://www.rainviewer.com/api.html), which provides worldwide radar composite data. No API key is needed. RainViewer's terms specify personal/educational use with attribution.
