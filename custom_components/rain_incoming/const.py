DOMAIN = "rain_incoming"

# Config keys (user-facing)
CONF_LOCATION_NAME = "location_name"
CONF_LOOKAHEAD_MINUTES = "lookahead_minutes"
CONF_MAP_STYLE = "map_style"

# Location name display limit (chars). Longer names overlap the attribution line.
MAX_LOCATION_NAME_CHARS = 30

# Maximum number of rain_incoming locations a user can configure.
# RainViewer's free tier rate-limits parallel tile fetching across multiple
# locations. Empirically, 4 locations is the practical maximum without
# degraded performance (5+ second slow-tile-fetch warnings start appearing).
# This limit can be raised when #87 (shared global tile cache) lands and
# reduces per-location fetch cost.
MAX_LOCATIONS = 4

# Lookahead bounds
DEFAULT_LOOKAHEAD_MINUTES = 60
MIN_LOOKAHEAD_MINUTES = 20
MAX_LOOKAHEAD_MINUTES = 60

# System config - not user-configurable
POLL_INTERVAL_SECONDS = 600
BACKOFF_BASE_SECONDS = 5
BACKOFF_MAX_SECONDS = 60
BACKOFF_MULTIPLIER = 2

# Frame thresholds
MIN_FRAMES_NORMAL = 3
MIN_FRAMES_DEGRADED = 2

# Detection thresholds
INTENSITY_THRESHOLD = 0.1
MIN_CELL_AREA_PIXELS = 4
MIN_TEMPORAL_FRAMES = 2
MAX_ANGULAR_VARIANCE_RADIANS = 0.5
MAX_STORM_SPEED_KMH = 120.0
PROXIMITY_RADIUS_KM = 15.0

# Radar image config
RADAR_RADII_KM = [64, 128, 256]
RADAR_GIF_FRAME_DURATION_MS = 500

# RainViewer tile config
RAINVIEWER_ZOOM = 7
RAINVIEWER_TILE_SIZE = 256
RAINVIEWER_COLOUR_SCHEME = 6
RAINVIEWER_ANALYSIS_GRID = 2  # fetch (2*N+1)^2 tiles centred on location

# Required attribution for RainViewer tile data per their API terms.
# rainviewer.com/api.html: "We kindly ask you to mention the Rain Viewer API
# as a source of the data on your website with a link: https://www.rainviewer.com/"
# We can't render hyperlinks in a baked GIF, so we include the domain in the text.
RAINVIEWER_ATTRIBUTION = "rainviewer.com"
