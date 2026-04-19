FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary local dirs
RUN mkdir -p /app/local_data

COPY . /app/

# Define an entrypoint that allows easy CLI execution
ENTRYPOINT ["make"]
CMD ["help"]
