FROM python:3.10-slim


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libpq-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy dependency list and install Python packages
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]