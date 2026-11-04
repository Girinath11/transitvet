# TransitVet in a container:
#   docker build -t transitvet .
#   docker run -p 8501:8501 transitvet                      (dashboard on http://localhost:8501)
#   docker run -v "$PWD/out:/out" transitvet python -m transitvet run "TIC 1333933" --out /out
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY . .
RUN python -m pytest -q -m "not network"
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true"]
