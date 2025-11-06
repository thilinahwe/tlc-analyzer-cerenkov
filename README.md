# tlc-analyzer-cerenkov
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


docker run --rm -it \
  -p 5173:5173 \
  -v "$PWD/frontend":/app \
  -w /app \
  -e CHOKIDAR_USEPOLLING=true \
  node:20-alpine \
  sh -lc 'npm install && npm run dev -- --host 0.0.0.0 --port 5173'