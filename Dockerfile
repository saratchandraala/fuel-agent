FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY fuel_data.csv .

# Expose port (Railway will set PORT env var)
EXPOSE 8000

# Run the application - Python will read PORT from environment
CMD ["python", "app.py"]
