import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from src.precedentiq.config import settings

if __name__ == "__main__":
    print(f"\n=======================================================")
    print(f" PrecedentIQ Backend is Running!")
    print(f" -> Local API:  http://localhost:{settings.API_PORT}")
    print(f" -> Swagger UI: http://localhost:{settings.API_PORT}/docs")
    print(f"=======================================================\n")
    uvicorn.run(
        "src.precedentiq.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
