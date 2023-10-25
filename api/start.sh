#!/bin/sh

streamlit run your_app.py --server.port=${PORT:-8080}
chmod +x start.sh
