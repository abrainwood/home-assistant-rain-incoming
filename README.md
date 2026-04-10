# Incoming Rain

**Know before it hits.** Incoming Rain watches weather radar and tells you when rain is heading your way - before it arrives. Get animated radar maps and sensors you can use to trigger automations.

![Radar showing heavy rain approaching Vancouver](docs/radar_vancouver_128km.gif)

*Heavy rain approaching Vancouver - 128km view*

---

## 🌧️ What you get

- **Binary sensor** - is rain incoming? (`on`/`off`, use it in any automation)
- **Arrival time prediction** - when will it arrive?
- **Intensity level** - light / moderate / heavy / extreme
- **Animated radar maps** at 4 zoom levels (64 / 128 / 256 / 512km)
- **Multi-location support** - add as many instances as you need
- **Works worldwide** - powered by [RainViewer](https://www.rainviewer.com/), free, no API key needed
- **Smart noise filtering** - QC pipeline handles radar artifacts so you don't get false alarms

---

## 📦 Installation

### Method 1: HACS (recommended)

1. Make sure [HACS](https://hacs.xyz) is installed
2. Go to **HACS > Integrations > Explore & Download Repositories**
3. Search for **"Incoming Rain"**
4. Click **Download**
5. Restart Home Assistant
6. Go to **Settings > Integrations > Add Integration** and search for **"Incoming Rain"**

### Method 2: Manual

1. Download the latest release from [GitHub](https://github.com/abrainwood/home-assistant-incoming-rain/releases)
2. Copy the `custom_components/incoming_rain` folder into your `config/custom_components/` directory
3. Restart Home Assistant
4. Go to **Settings > Integrations > Add Integration** and search for **"Incoming Rain"**

---

## ⚙️ Configuration

No YAML needed - everything is configured through the Home Assistant UI.

The latitude and longitude default to your Home Assistant home location (set in **Settings > General**), so for most people you just hit Save and you're done.

| Option | Default | Range | Description |
|---|---|---|---|
| Latitude | Your HA home location | -90 to 90 | Location to monitor |
| Longitude | Your HA home location | -180 to 180 | Location to monitor |
| Lookahead | 60 min | 15-120 min | How far ahead to predict |

### Changing settings after setup

**Settings > Integrations > Incoming Rain > Configure**

Changes take effect immediately on the next poll.

---

## 💡 What can you do with it?

- Close a pergola cover before rain arrives
- Send a notification to bring the washing in
- Return a robot lawnmower to base before it gets soaked
- Close a pool cover
- Pause or cancel an irrigation schedule
- Alert if rain is expected during an outdoor event
- Close skylights or windows (with smart window actuators)

---

## 🔧 Example automations

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

### Bring the washing in

```yaml
automation:
  - alias: "Washing alert - rain incoming"
    trigger:
      - platform: state
        entity_id: binary_sensor.incoming_rain_status
        to: "on"
    condition:
      # Only alert during daytime when washing might be out
      - condition: time
        after: "07:00:00"
        before: "19:00:00"
    action:
      - service: notify.mobile_app_your_phone
        data:
          title: "Rain incoming!"
          message: >
            Rain expected to arrive at {{ states('sensor.incoming_rain_arrival_time') | as_timestamp | timestamp_custom('%H:%M') }}.
            Intensity: {{ states('sensor.incoming_rain_intensity') }}.
            Time to bring the washing in!
```

---

## 📋 Entity reference

| Entity | Type | Description |
|---|---|---|
| `binary_sensor.incoming_rain_status` | Binary | `on` when rain is approaching or overhead |
| `sensor.incoming_rain_arrival_time` | Timestamp | Predicted arrival time, `unknown` when no rain incoming |
| `sensor.incoming_rain_intensity` | Sensor | Precipitation intensity: none / light / moderate / heavy / extreme |
| `sensor.incoming_rain_last_rain` | Timestamp | Last time rain was detected nearby |
| `image.incoming_rain_radar_64km` | Image | Animated radar map - neighbourhood scale (64km radius) |
| `image.incoming_rain_radar_128km` | Image | Animated radar map - city/regional scale (128km radius) |
| `image.incoming_rain_radar_256km` | Image | Animated radar map - state/province scale (256km radius) |
| `image.incoming_rain_radar_512km` | Image | Animated radar map - synoptic scale (512km radius) |

The radar images show the last several frames of radar data as an animation, with a crosshair marking your monitored location.

### Sensor attributes

The binary sensor and arrival time sensor expose these attributes for debugging and automation:

| Attribute | Description |
|---|---|
| `confidence` | `normal` (3+ frames), `degraded` (2 frames), or `unavailable` |
| `frame_count` | Number of radar frames used in the analysis |

---

## 🔍 Troubleshooting

### Confidence levels

- **normal** - 3 or more radar frames available. Detection is running at full accuracy.
- **degraded** - only 2 frames available. Detection still works but motion tracking is less reliable. This can happen when the RainViewer API is temporarily returning fewer frames.
- **unavailable** - fewer than 2 frames. Sensors will show as unavailable until more data arrives.

### Cold-start period

The integration learns your local clutter patterns over 12+ hours. During this period you may see slightly reduced accuracy.

### False positives

Anomalous propagation (AP) is a radar artifact that can appear during clear, still nights. The QC system progressively filters these as it learns your local patterns.

### No rain detected when it's clearly raining

Check that your configured latitude and longitude are correct. The crosshair on the radar image entities shows the monitored location - verify it matches where you expect.

---

## ⚙️ How it works

Incoming Rain fetches radar composite data from RainViewer and analyses multiple frames to track rain cell motion and project whether any cells will reach your location within the lookahead window. A QC pipeline filters out radar noise - things like ground clutter and anomalous propagation - using spatial, temporal, and directional coherence checks. Rain is reported as incoming if it's overhead or projected to arrive within your configured window.

For technical details, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 🤝 Contributing

Bug reports, feature requests, and PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a dev environment and run the test suite.

---

## Requirements

- Home Assistant 2024.1+
- Internet access (RainViewer API - free, no API key required)

---

## Data source

Radar data provided by the [RainViewer API](https://www.rainviewer.com/api.html) - worldwide radar composites, free for personal/educational use with attribution.

## License

[MIT](LICENSE)
