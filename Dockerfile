# Sử dụng Python base image
FROM python:3.9-slim

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Cài đặt các dependencies hệ thống
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt vào container
COPY requirements.txt /app/requirements.txt

# Cài đặt các dependencies Python
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy mã nguồn vào container
COPY . /app

# Mở cổng cho ứng dụng
EXPOSE 8080

# Lệnh để chạy ứng dụng khi container khởi động
CMD ["uvicorn", "api_yolo:app", "--host", "0.0.0.0", "--port", "8080"]
