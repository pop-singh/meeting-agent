#!/bin/bash

uvicorn backend:app --host 0.0.0.0 --port 8000 &
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
wait
