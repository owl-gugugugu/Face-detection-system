from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.core.backgroundThread import BackgroundThread
from backend.core.camera import get_camera
from backend.core.face_engine import get_face_engine
from backend.routers import auth, face, stream, unlock
from backend.database.manager import db_manager as db
from backend.config import validate_config, print_config_summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时验证配置
    print("\n" + "="*60)
    print("智能门禁系统启动中...")
    print("="*60)

    # 验证配置
    is_valid = validate_config()
    if not is_valid:
        print("[ERROR] 配置验证失败，请检查 config.py")
        raise RuntimeError("配置验证失败")

    # 打印配置摘要
    print_config_summary()

    # 引擎初始化以及摄像头初始化
    print("正在初始化人脸识别引擎...")
    face_engine = get_face_engine()

    print("正在初始化摄像头...")
    camera = get_camera()

    print("正在启动后台守护线程...")
    back = BackgroundThread()
    back.start()

    print("\n[OK] 智能门禁系统启动成功！")
    print("="*60 + "\n")

    yield

    # 引擎销毁以及摄像头销毁
    print("\n智能门禁系统正在关闭...")
    back.stop()
    db.close()  # 关闭数据库连接
    del face_engine
    del camera
    print("[OK] 系统已关闭\n")


app = FastAPI(lifespan=lifespan)
app.include_router(auth.router)
app.include_router(face.router)
app.include_router(stream.router)
app.include_router(unlock.router)
