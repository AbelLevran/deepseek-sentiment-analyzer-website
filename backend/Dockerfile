FROM python:3.9-slim

# Buat user non-root khusus untuk Hugging Face Spaces (wajib)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Salin file proyek ke dalam container dengan hak akses untuk user
COPY --chown=user . /app

# Instal pustaka yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt

# Pra-unduh NLTK data ke direktori yang benar agar tidak error saat runtime
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/home/user/nltk_data'); nltk.download('punkt', download_dir='/home/user/nltk_data')"
ENV NLTK_DATA="/home/user/nltk_data"

# Ekspos port Hugging Face
EXPOSE 7860

# Jalankan server API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
