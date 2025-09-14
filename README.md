# Webfloox-alexandre
Here are the steps to create the pgsql database with docker :
First, pull the postgres docker image :
```bash
docker pull postgres
```
Then launch the database :
```bash
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=mypassword \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=webfloox \
  -p 5432:5432 \
  -v postgres-data:/var/lib/postgresql/data \
  -d postgres
```
