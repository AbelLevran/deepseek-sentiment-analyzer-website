# 1. Gunakan image Python yang ringan
FROM python:3.9-slim

# 2. Set working directory di dalam container
WORKDIR /app

# 3. Salin semua file dari folder backend saat ini ke dalam folder /app di container
COPY . /app

# 4. Instal semua pustaka yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# 5. Hugging Face Spaces secara otomatis mengekspos port 7860
EXPOSE 7860

# 6. Jalankan FastAPI menggunakan uvicorn di port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
