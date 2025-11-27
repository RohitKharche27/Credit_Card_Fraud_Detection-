# Dockerfile for Streamlit app
FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app


ENV MODEL_PATH=/app/model/fraud_model.pkl
EXPOSE 8501


CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.enableCORS=false"]
