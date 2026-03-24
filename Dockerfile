FROM python:3.10-slim
ARG RUN_ID
ENV MODEL_DIR=/opt/ml/model
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p ${MODEL_DIR}
RUN echo "Downloading model for Run ID: ${RUN_ID}" > ${MODEL_DIR}/run_id.txt
WORKDIR /app
COPY src/ ./
CMD ["python", "app.py"]
