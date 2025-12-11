# Backend Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# not to buffer output so that continuous logging is possible
ENV PYTHONUNBUFFERED=1
# tells readers which port(in the container's world) this app is expected to listen on.
# even without this line the conatiner app runs fine
EXPOSE 8000 
# Run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
