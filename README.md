## Magisterka — integracja z Ollama

W projekcie backend Spring Boot korzysta z lokalnego modelu językowego przez API Ollama. 
Początkowo planowane było uruchomienie Ollamy jako osobnego serwisu w `docker-compose.yml`, np. pod adresem:
SPRING_OLLAMA_URL=http://ollama:11434
występował błąd
failed to copy: httpReadSeeker: failed open: failed to do request:
Get "https://production.cloudfront.docker.com/..."
EOF

Ostatecznie Ollama nie jest uruchamiana jako kontener Docker, ponieważ pobieranie obrazu ollama/ollama kończyło się błędem EOF. 
Zamiast tego Ollama działa lokalnie na Windowsie, a aplikacja Spring Boot uruchomiona w Dockerze komunikuje się z nią przez:
http://host.docker.internal:11434

Takie rozwiązanie pozwala zachować konteneryzację pozostałych usług, takich jak PostgreSQL,
bez konieczności pobierania dużego obrazu Dockerowego Ollamy.