use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Instant,
};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use uuid::Uuid;

#[derive(Debug, Default)]
struct Stats {
    total_ops: u64,
    detect_ops: u64,
    track_ops: u64,
    heatmap_ops: u64,
    scene_ops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BoundingBox {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

#[derive(Debug, Clone)]
struct TrackedObject {
    track_id: String,
    bbox: BoundingBox,
    class: String,
    age_frames: u32,
}

#[derive(Debug, Default)]
struct AppData {
    stats: Stats,
    tracks: HashMap<String, TrackedObject>, // track_id -> obj
    heatmap_accumulator: Vec<Vec<f32>>,     // 32x32 grid
}

type AppState = Arc<Mutex<AppData>>;

// --- /api/v1/va/detect ---
#[derive(Debug, Deserialize)]
struct DetectRequest {
    frame_id: String,
    width: u32,
    height: u32,
    // base64 encoded or metadata-only mode
    #[serde(default)]
    classes_filter: Vec<String>,
    #[serde(default = "default_conf")]
    confidence_threshold: f32,
}

fn default_conf() -> f32 {
    0.5
}

#[derive(Debug, Serialize)]
struct Detection {
    detection_id: String,
    class: String,
    confidence: f32,
    bbox: BoundingBox,
}

#[derive(Debug, Serialize)]
struct DetectResponse {
    request_id: String,
    frame_id: String,
    detections: Vec<Detection>,
    inference_ms: f32,
    model: String,
}

async fn detect_objects(
    State(state): State<AppState>,
    Json(req): Json<DetectRequest>,
) -> Result<Json<DetectResponse>, StatusCode> {
    {
        let mut d = state.lock().unwrap();
        d.stats.total_ops += 1;
        d.stats.detect_ops += 1;
    }

    // Simulated detections based on frame metadata
    let mut detections = Vec::new();
    let classes = if req.classes_filter.is_empty() {
        vec!["person", "vehicle", "bicycle"]
    } else {
        req.classes_filter.iter().map(|s| s.as_str()).collect()
    };

    for (i, class) in classes.iter().enumerate() {
        let conf = 0.85 - i as f32 * 0.1;
        if conf >= req.confidence_threshold {
            detections.push(Detection {
                detection_id: Uuid::new_v4().to_string(),
                class: class.to_string(),
                confidence: conf,
                bbox: BoundingBox {
                    x: (req.width as f32 * 0.1 * (i + 1) as f32).min(req.width as f32 - 50.0),
                    y: (req.height as f32 * 0.15 * (i + 1) as f32).min(req.height as f32 - 50.0),
                    w: 80.0,
                    h: 120.0,
                },
            });
        }
    }

    Ok(Json(DetectResponse {
        request_id: Uuid::new_v4().to_string(),
        frame_id: req.frame_id,
        detections,
        inference_ms: 12.4,
        model: "alice-det-v1".to_string(),
    }))
}

// --- /api/v1/va/track ---
#[derive(Debug, Deserialize)]
struct TrackRequest {
    frame_id: String,
    detections: Vec<TrackDetection>,
}

#[derive(Debug, Deserialize)]
struct TrackDetection {
    class: String,
    confidence: f32,
    bbox: BoundingBox,
}

#[derive(Debug, Serialize)]
struct TrackEntry {
    track_id: String,
    class: String,
    confidence: f32,
    bbox: BoundingBox,
    age_frames: u32,
    status: String,
}

#[derive(Debug, Serialize)]
struct TrackResponse {
    request_id: String,
    frame_id: String,
    tracks: Vec<TrackEntry>,
    active_tracks: usize,
}

async fn track_objects(
    State(state): State<AppState>,
    Json(req): Json<TrackRequest>,
) -> Result<Json<TrackResponse>, StatusCode> {
    let mut d = state.lock().unwrap();
    d.stats.total_ops += 1;
    d.stats.track_ops += 1;

    // simple IoU-based association (stub: always assign new track ids for new detections)
    let mut tracks = Vec::new();
    for det in &req.detections {
        // find existing track within IoU threshold
        let matched = d.tracks.values_mut().find(|t| {
            let iou_approx = {
                let dx = (t.bbox.x - det.bbox.x).abs();
                let dy = (t.bbox.y - det.bbox.y).abs();
                dx < 60.0 && dy < 80.0 && t.class == det.class
            };
            iou_approx
        });

        if let Some(t) = matched {
            t.bbox = det.bbox.clone();
            t.age_frames += 1;
            tracks.push(TrackEntry {
                track_id: t.track_id.clone(),
                class: t.class.clone(),
                confidence: det.confidence,
                bbox: det.bbox.clone(),
                age_frames: t.age_frames,
                status: "tracked".to_string(),
            });
        } else {
            let tid = Uuid::new_v4().to_string();
            let obj = TrackedObject {
                track_id: tid.clone(),
                bbox: det.bbox.clone(),
                class: det.class.clone(),
                age_frames: 1,
            };
            tracks.push(TrackEntry {
                track_id: tid.clone(),
                class: det.class.clone(),
                confidence: det.confidence,
                bbox: det.bbox.clone(),
                age_frames: 1,
                status: "new".to_string(),
            });
            d.tracks.insert(tid, obj);
        }
    }

    let active = d.tracks.len();
    Ok(Json(TrackResponse {
        request_id: Uuid::new_v4().to_string(),
        frame_id: req.frame_id,
        tracks,
        active_tracks: active,
    }))
}

// --- /api/v1/va/heatmap ---
#[derive(Debug, Deserialize)]
struct HeatmapRequest {
    points: Vec<[f32; 2]>, // normalized 0-1 coords
    #[serde(default = "default_grid")]
    grid_size: usize,
    #[serde(default)]
    decay: f32,
}

fn default_grid() -> usize {
    32
}

#[derive(Debug, Serialize)]
struct HeatmapResponse {
    request_id: String,
    grid_size: usize,
    grid: Vec<Vec<f32>>,
    peak_x: usize,
    peak_y: usize,
    peak_value: f32,
    total_points: usize,
}

async fn generate_heatmap(
    State(state): State<AppState>,
    Json(req): Json<HeatmapRequest>,
) -> Result<Json<HeatmapResponse>, StatusCode> {
    let gs = req.grid_size.clamp(8, 128);
    let mut grid = vec![vec![0.0f32; gs]; gs];

    // gaussian splat each point
    let sigma = gs as f32 * 0.08;
    for pt in &req.points {
        let cx = (pt[0].clamp(0.0, 1.0) * (gs - 1) as f32) as usize;
        let cy = (pt[1].clamp(0.0, 1.0) * (gs - 1) as f32) as usize;
        for y in 0..gs {
            for x in 0..gs {
                let dx = x as f32 - cx as f32;
                let dy = y as f32 - cy as f32;
                let g = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                grid[y][x] += g;
            }
        }
    }

    // apply decay to accumulator
    let decay = req.decay.clamp(0.0, 1.0);
    {
        let mut d = state.lock().unwrap();
        d.stats.total_ops += 1;
        d.stats.heatmap_ops += 1;
        if d.heatmap_accumulator.len() == gs {
            for y in 0..gs {
                for x in 0..gs {
                    d.heatmap_accumulator[y][x] = d.heatmap_accumulator[y][x] * decay + grid[y][x];
                    grid[y][x] = d.heatmap_accumulator[y][x];
                }
            }
        } else {
            d.heatmap_accumulator = grid.clone();
        }
    }

    // find peak
    let (mut px, mut py, mut pv) = (0, 0, 0.0f32);
    for y in 0..gs {
        for x in 0..gs {
            if grid[y][x] > pv {
                pv = grid[y][x];
                px = x;
                py = y;
            }
        }
    }

    // normalize
    if pv > 0.0 {
        for row in &mut grid {
            for v in row.iter_mut() {
                *v = (*v / pv * 1000.0).round() / 1000.0;
            }
        }
    }

    Ok(Json(HeatmapResponse {
        request_id: Uuid::new_v4().to_string(),
        grid_size: gs,
        grid,
        peak_x: px,
        peak_y: py,
        peak_value: (pv * 1000.0).round() / 1000.0,
        total_points: req.points.len(),
    }))
}

// --- /api/v1/va/scenes ---
#[derive(Debug, Deserialize)]
struct ScenesRequest {
    frame_id: String,
    width: u32,
    height: u32,
    #[serde(default)]
    context_tags: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SceneClass {
    label: String,
    confidence: f32,
}

#[derive(Debug, Serialize)]
struct ScenesResponse {
    request_id: String,
    frame_id: String,
    scene_classes: Vec<SceneClass>,
    dominant_scene: String,
    activity_level: String,
    crowd_density: f32,
}

async fn classify_scenes(
    State(state): State<AppState>,
    Json(req): Json<ScenesRequest>,
) -> Result<Json<ScenesResponse>, StatusCode> {
    {
        let mut d = state.lock().unwrap();
        d.stats.total_ops += 1;
        d.stats.scene_ops += 1;
    }

    // deterministic mock based on resolution heuristics
    let is_outdoor = req.width > 1280 || req.context_tags.iter().any(|t| t == "outdoor");
    let scene_classes = if is_outdoor {
        vec![
            SceneClass { label: "outdoor_street".to_string(), confidence: 0.82 },
            SceneClass { label: "crowd".to_string(), confidence: 0.64 },
            SceneClass { label: "vehicle_traffic".to_string(), confidence: 0.51 },
        ]
    } else {
        vec![
            SceneClass { label: "indoor_office".to_string(), confidence: 0.78 },
            SceneClass { label: "meeting_room".to_string(), confidence: 0.55 },
            SceneClass { label: "corridor".to_string(), confidence: 0.41 },
        ]
    };

    let dominant = scene_classes[0].label.clone();
    let crowd_density = if is_outdoor { 0.45 } else { 0.18 };
    let activity_level = if crowd_density > 0.4 { "high" } else { "low" }.to_string();

    Ok(Json(ScenesResponse {
        request_id: Uuid::new_v4().to_string(),
        frame_id: req.frame_id,
        scene_classes,
        dominant_scene: dominant,
        activity_level,
        crowd_density,
    }))
}

// --- /api/v1/va/stats ---
#[derive(Debug, Serialize)]
struct StatsResponse {
    service: &'static str,
    version: &'static str,
    total_ops: u64,
    detect_ops: u64,
    track_ops: u64,
    heatmap_ops: u64,
    scene_ops: u64,
    active_tracks: usize,
}

async fn get_stats(State(state): State<AppState>) -> Json<StatsResponse> {
    let d = state.lock().unwrap();
    Json(StatsResponse {
        service: "alice-video-analytics-core",
        version: env!("CARGO_PKG_VERSION"),
        total_ops: d.stats.total_ops,
        detect_ops: d.stats.detect_ops,
        track_ops: d.stats.track_ops,
        heatmap_ops: d.stats.heatmap_ops,
        scene_ops: d.stats.scene_ops,
        active_tracks: d.tracks.len(),
    })
}

// --- /health ---
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
    version: &'static str,
    uptime_secs: u64,
    total_ops: u64,
}

async fn health(
    State(state): State<AppState>,
    axum::extract::Extension(start): axum::extract::Extension<Arc<Instant>>,
) -> Json<HealthResponse> {
    let d = state.lock().unwrap();
    Json(HealthResponse {
        status: "ok",
        version: env!("CARGO_PKG_VERSION"),
        uptime_secs: start.elapsed().as_secs(),
        total_ops: d.stats.total_ops,
    })
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let state: AppState = Arc::new(Mutex::new(AppData::default()));
    let start = Arc::new(Instant::now());

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/v1/va/detect", post(detect_objects))
        .route("/api/v1/va/track", post(track_objects))
        .route("/api/v1/va/heatmap", post(generate_heatmap))
        .route("/api/v1/va/scenes", post(classify_scenes))
        .route("/api/v1/va/stats", get(get_stats))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .layer(axum::extract::Extension(start))
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8128);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("alice-video-analytics-core listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
