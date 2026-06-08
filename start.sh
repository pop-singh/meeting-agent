#!/bin/bash

# Start FastAPI in background (port 8000 internally)
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit (this uses Railway's assigned port)
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT

# Keep process alive
wait