from fastapi import APIRouter

from backend.core.doorController import get_door_controller

router = APIRouter(
    prefix="/api",
    tags=["moniter"],
    responses={404: {"description": "Not found"}},
)

@router.post("/unlock")
async def moniter():
    doorLock = get_door_controller()
    doorLock.open()
    return {"status": "success"}
