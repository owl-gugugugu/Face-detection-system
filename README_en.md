# SmartGate-App (Intelligent Access Control System)

A smart face recognition access control system based on the RK3568 platform. It adopts an architecture combining FastAPI, a Web Frontend, and a C++ Inference Engine, supporting both local development debugging and embedded deployment.

The inference model utilizes the "RetinaFace+MobileFaceNet" industrial gold-standard configuration for face detection and recognition.

## ✨ Features

- 🔐 **Face Recognition Access Control** - High-precision face recognition based on RetinaFace + MobileFaceNet.
- 📱 **Web Management Interface** - Manage directly via mobile/PC browsers without installing an App.
- 🎥 **Real-time Video Stream** - MJPEG real-time camera preview.
- 👤 **User Management** - Face enrollment, deletion, and list viewing.
- 🚪 **Remote Unlocking** - One-click remote door opening via the Web interface.
- 🔑 **Password Management** - Administrator login authentication and password modification.
- 🖥️ **Development Mode** - Mock hardware simulation supporting PC development and debugging.
- ⚡ **NPU Acceleration** - Runs on the NPU using the RKNN SDK for low power consumption and high performance.

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python Web framework.
- **OpenCV** - Image processing and video streaming.
- **SQLite** - Lightweight database.
- **PyJWT** - JWT authentication.
- **Uvicorn** - ASGI server.

### Frontend
- **Bootstrap 5** - Responsive UI framework.
- **Vanilla JavaScript** - Native JS, no framework dependencies.
- **Jinja2** - Server-side template engine.

### AI Inference Engine
- **RetinaFace** - Face detection model.
- **MobileFaceNet** - Face feature extraction.
- **RKNN SDK** - Rockchip NPU inference framework.

## 📁 Project Structure

```
SmartGate-App/
├── backend/                # Backend Service
│   ├── main.py            # FastAPI application entry point
│   ├── config.py          # Configuration file (Important)
│   ├── start_server.py    # Startup script
│   ├── routers/           # API Routers
│   │   ├── auth.py        # Authentication interfaces
│   │   ├── face.py        # Face management interfaces
│   │   ├── unlock.py      # Door control interfaces
│   │   ├── stream.py      # Video stream interfaces
│   │   └── pages.py       # Page routes
│   ├── core/              # Core Functionality
│   │   ├── camera.py      # Camera management
│   │   ├── face_engine.py # Face engine wrapper
│   │   ├── doorController.py  # Door lock control
│   │   ├── backgroundThread.py # Background daemon thread
│   │   └── mock.py        # Mock simulation (For development)
│   ├── database/          # Database
│   │   └── manager.py     # Database manager
│   └── utils/             # Utility Functions
│       ├── auth.py        # JWT authentication
│       └── password.py    # Password tools
├── fronted/               # Frontend Pages
│   ├── templates/         # HTML Templates
│   │   ├── login.html     # Login page
│   │   ├── dashboard.html # Main dashboard
│   │   ├── face_input.html    # Face enrollment
│   │   └── face_list.html     # User list
│   └── static/            # Static Resources
│       ├── css/           # Stylesheets
│       └── js/            # JavaScript
├── face_app/              # C++ Inference Engine
│   ├── lib/               # Compiled .so libraries
│   └── include/           # Header files
├── face_detection/        # Face detection model files
└── docs/                  # Documentation
    └── 前后端集成文档.md  # Detailed technical documentation
```

## 🚀 Quick Start

### Development Environment (PC)

**1. Install Dependencies**
```bash
pip install -r backend/requirements.txt
```

**2. Configure Development Mode**

Ensure `backend/config.py` is set to development mode:
```python
DEV_MODE = True  # Use Mock hardware simulation
```

**3. Start Server**
```bash
python backend/start_server.py
```

**4. Access System**

```
Open in browser: http://localhost:8000
Default Account: admin
Default Password: 123456
```

### Production Environment (RK3568)

**1. Modify Configuration**

Edit `backend/config.py`:
```python
DEV_MODE = False  # Switch to production mode
GPIO_DOOR_PIN = 17  # Configure door lock GPIO pin
```

**2. Hardware Connection**
- Connect the OV5695 camera to the CSI interface.
- Connect the door lock relay to GPIO17 (or a custom pin).
- Ensure hardware power supply is normal.

**3. Install System Dependencies**
```bash
# Install necessary system libraries on RK3568
sudo apt-get install python3-opencv
```

**4. Start Service**
```bash
python backend/start_server.py
```

**5. Configure Auto-start** (Optional)
```bash
# Create systemd service file
sudo nano /etc/systemd/system/smartgate.service
```

## 💡 Core Functions Explained

### 1. Face Recognition Flow

```
Camera Capture → Face Detection (RetinaFace) → Feature Extraction (MobileFaceNet)
→ Feature Comparison (Cosine Similarity) → Identity Verification → Unlock/Reject
```

### 2. Dual-Mode Architecture

| Mode | Description | Hardware Dependency | Usage Scenario |
|------|-------------|---------------------|----------------|
| Development Mode | Mock Simulation | None | PC Development & Debugging |
| Production Mode | Real Hardware | RK3568 + Camera | Actual Deployment |

### 3. API Interfaces

Complete RESTful API supporting:
- JWT Authentication & Authorization
- Face Enrollment/Deletion/Query
- Real-time Video Stream (MJPEG)
- Remote Door Control
- Password Management

See details: [Frontend-Backend Integration Docs](docs/前后端集成文档.md)

## 🧠 Face Recognition Module Details

### Architecture Design

This system adopts the **RetinaFace + MobileFaceNet** industrial-grade face recognition solution:

```bash
JPEG Image (Any Size)
    ↓
[1. Image Decode] → cv::Mat (BGR)
    ↓
[2. Resize] → 640×640
    ↓
[3. RetinaFace Inference] → Face Box + 5 Keypoints (NPU Accelerated)
    ↓
[4. Face Alignment] → Affine Transformation → 112×112 RGB (CPU)
    ↓
[5. MobileFaceNet Inference] → 512-dim Feature Vector (NPU Accelerated, L2 Normalized)
    ↓
[6. Cosine Similarity Calculation] → Match Result (0~1)
```

### Performance Metrics

**Runtime Performance (RK3568 NPU)**

| Stage | Time Cost | Description |
|-------|-----------|-------------|
| Image Decode + Preprocess | ~20ms | OpenCV imdecode + resize |
| RetinaFace Inference | ~40ms | NPU Hardware Acceleration |
| Face Alignment | ~5ms | Affine Transformation (CPU) |
| MobileFaceNet Inference | ~30ms | NPU Hardware Acceleration |
| Feature Comparison | < 1ms | Cosine Similarity Calculation |
| **Total Single Recognition** | **~96ms** | Approx. 10 FPS |

**Model Specifications**

| Model | Input Size | Output | Accuracy | File Size |
|-------|------------|--------|----------|-----------|
| RetinaFace | 640×640 RGB | Face Box + 5 Keypoints | 95%+ (WIDER FACE) | ~2.5MB |
| MobileFaceNet | 112×112 RGB | 512-dim Feature Vector | 99.5%+ (LFW) | ~4MB |

### Key Configuration Parameters

**Recommended Similarity Thresholds**

| Scenario | Recommended Threshold | Description |
|----------|-----------------------|-------------|
| Security (Payment, Access) | **0.7** | Strict mode, low false acceptance rate |
| General (Attendance, Album) | **0.6** | Recommended value, balance between accuracy and convenience |
| Loose (Recommender Systems) | **0.5** | High recall rate |

> **Note**: The current system defaults to 0.5. You can modify `SIMILARITY_THRESHOLD` in `backend/config.py`.

**Detection Parameters**

```c
CONF_THRESHOLD = 0.5    // Face confidence threshold
NMS_THRESHOLD = 0.4     // NMS IoU threshold
MIN_FACE_SIZE = 40      // Minimum face size (pixels)
```

### Module Architecture

```bash
┌─────────────────────────────────┐
│  Layer 3: FastAPI Router Layer  │  ← Business Logic
│  - HTTP Interface                │
│  - Database Operations           │
│  - JWT Authentication            │
└─────────────────────────────────┘
              ↓ Calls
┌─────────────────────────────────┐
│  Layer 2: Python Wrapper        │  ← Interface Encapsulation
│  (backend/core/face_engine.py)  │
│  - ctypes binding to C++ library │
│  - Singleton Pattern Management  │
└─────────────────────────────────┘
              ↓ ctypes
┌─────────────────────────────────┐
│  Layer 1: C++ Dynamic Library   │  ← Core Computation
│  (libface_engine.so)            │
│  - RKNN Model Inference          │
│  - OpenCV Image Processing       │
│  - Face Alignment Algorithms     │
└─────────────────────────────────┘
```

### Usage Example

**Python Interface**

```python
from backend.core.face_engine import get_face_engine

# Get engine instance (Singleton)
engine = get_face_engine()

# Extract face features
with open("face1.jpg", "rb") as f:
    feature1 = engine.extract_feature(f.read())

with open("face2.jpg", "rb") as f:
    feature2 = engine.extract_feature(f.read())

# Compute similarity
if feature1 and feature2:
    similarity = engine.compute_similarity(feature1, feature2)
    print(f"Similarity: {similarity:.4f}")

    if similarity >= 0.6:
        print("✓ Same person")
    else:
        print("✗ Different person")
```

### Technical Advantages

- ✅ **NPU Hardware Acceleration** - Uses RKNN SDK for fast inference and low power consumption.
- ✅ **Multi-scale Detection** - RetinaFace supports detecting small faces over 40px.
- ✅ **High Precision Recognition** - MobileFaceNet achieves 99.5%+ accuracy on the LFW dataset.
- ✅ **Lightweight Deployment** - Total model size is only 6.5MB, dynamic library 15MB.
- ✅ **Industrial-grade Architecture** - C++ Core + Python Wrapper, stable and reliable.

**Detailed Technical Documentation**: [face_detection/README.md](face_detection/README.md)

## 📚 Documentation

- [Frontend-Backend Integration Docs](docs/前后端集成文档.md) - API interfaces, deployment guide, FAQ.
- [Face Recognition Module Docs](face_detection/README.md) - Architecture design, compilation, deployment, performance optimization.
- [Model Training Repository](https://github.com/JuyaoHuang/SmartGaze-model-zoo) - Training code for RetinaFace and MobileFaceNet.

## 🔧 Configuration Guide

Key configurations (`backend/config.py`):

```python
# Run Mode
DEV_MODE = True  # True=Dev Mode, False=Prod Mode

# JWT Authentication
JWT_SECRET_KEY = '123456'  # Please change for production
JWT_EXPIRE_HOURS = 24      # Token validity period

# Camera Config
CAMERA_INDEX = 0           # Camera index
CAMERA_WIDTH = 640         # Resolution Width
CAMERA_HEIGHT = 480        # Resolution Height

# Face Recognition
SIMILARITY_THRESHOLD = 0.5  # Similarity threshold (0-1)

# GPIO Config
GPIO_DOOR_PIN = None       # Door lock control pin (Set in Prod)

# Server
SERVER_HOST = '0.0.0.0'    # Listen address
SERVER_PORT = 8000         # Port number
```

## 🐛 FAQ

**Q: 422 Unprocessable Entity Error?**

A: Mismatched frontend/backend data formats. Ensure the backend uses Pydantic models to receive JSON.

**Q: Video stream shows a gray screen?**

A: In Development Mode, the Mock camera displays a gray screen with text. This is normal behavior.

**Q: Face recognition failed?

A: In Production Mode, check the camera connection and ensure the `.so` library is loaded correctly.

**Q: How to change the default password?**

A: Modify `DEFAULT_ADMIN_PASSWORD` in `config.py` and re-initialize the database.

For more questions: [Frontend-Backend Integration Docs - FAQ](docs/前后端集成文档.md#7-常见问题)

## 🤝 Contribution

Issues and Pull Requests are welcome!

## 📄 License

This project is a coursework project and is for learning and exchange purposes only.

## 👥 Authors

- [Juyao Huang](https://github.com/JuyaoHuang) - System Design, Development, and Review
- [Haoran Wu](https://github.com/owl-gugugugu) - Backend Development
- [Junshi Kong](https://github.com/kjs123456) - Frontend Development

## 🔗 Related Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RKNN Toolkit2 Documentation](https://github.com/rockchip-linux/rknn-toolkit2)
- [Bootstrap 5 Documentation](https://getbootstrap.com/)