# TODO: implement once the API and a trained model artifact exist.
#
# Sketch:
#
# FROM python:3.12-slim
# WORKDIR /app
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
# COPY src/ src/
# COPY pyproject.toml .
# RUN pip install --no-cache-dir -e .
# COPY models/ models/
# COPY frontend/ frontend/
# EXPOSE 8000
# CMD ["uvicorn", "paletteml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
#
# Note: pin the image to a Python version confirmed to have wheels
# for scikit-image/scikit-learn at build time (see README).
