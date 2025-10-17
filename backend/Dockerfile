FROM python:3.12.3

WORKDIR /backend

COPY ./backend/requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY ./backend/ .
COPY ./src/mind /src/mind
COPY .env . 
COPY ./data .

ENV PYTHONPATH=/backend:/src

EXPOSE 5001

CMD ["python3", "main.py"] 
