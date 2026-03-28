# ALICE-VideoAnalytics-SaaS

Video analytics API — part of the ALICE Eco-System.

## Overview

ALICE-VideoAnalytics-SaaS provides server-side video analytics primitives: object detection, multi-object tracking, heatmap generation, scene classification, and analytics statistics.

## Services

- **core-engine** — Detection, tracking, heatmap, scene analysis (port 8128)
- **api-gateway** — JWT auth, rate limiting, reverse proxy

## Quick Start

```bash
cd services/core-engine
cargo run

curl http://localhost:8128/health
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/v1/va/detect | Object detection in frame |
| POST | /api/v1/va/track | Multi-object tracking update |
| POST | /api/v1/va/heatmap | Generate presence heatmap |
| POST | /api/v1/va/scenes | Scene classification |
| GET  | /api/v1/va/stats | Service statistics |

## License

AGPL-3.0-or-later
