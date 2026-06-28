FROM maven:3.9.9-eclipse-temurin-21 AS build
WORKDIR /app

COPY pom.xml .
RUN mvn -B -DskipTests dependency:go-offline

# copy sources separately
COPY src ./src
# if you use them:
COPY mvnw ./
COPY .mvn ./.mvn

RUN mvn -DskipTests clean package

FROM eclipse-temurin:21-jre
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /app/venv

RUN /app/venv/bin/pip install --no-cache-dir \
    reportlab \
    matplotlib

WORKDIR /app
COPY --from=build /app/target/AIOmniHub-0.0.1-SNAPSHOT.jar /app/app.jar

COPY budowa_pdf.py /app/budowa_pdf.py
COPY DejaVuSans.ttf /app/DejaVuSans.ttf
EXPOSE 8080
CMD ["java","-jar","/app/app.jar"]
