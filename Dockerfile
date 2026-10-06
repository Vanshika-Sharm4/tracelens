# Stage 1: build the React UI
FROM node:20-slim AS ui
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python service (CPU torch keeps the image small; use a CUDA base image for GPU profiling)
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY tracelens ./tracelens
RUN pip install --no-cache-dir .
COPY --from=ui /app/frontend/dist ./frontend/dist
ENV TRACELENS_STATIC=/app/frontend/dist
EXPOSE 8000
CMD ["tracelens", "serve", "--host", "0.0.0.0", "--port", "8000"]
