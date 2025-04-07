# Stage 1: Build dependencies
FROM python:3.11 as builder

WORKDIR /app

# Copy only the pyproject.toml file
COPY pyproject.toml ./
# Create empty directories for the packages defined in pyproject.toml
RUN mkdir -p container_component_src pipeline
# Copy the README.md file that is referenced in pyproject.toml
COPY README.md ./

# Install dependencies only
RUN pip install --no-cache-dir .

# Stage 2: Final image
FROM python:3.11

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Now copy the actual code
COPY ./container_component_src /app/container_component_src
COPY ./pipeline /app/pipeline
COPY ./config.toml /app/config.toml
COPY ./README.md /app/
COPY pyproject.toml ./

# Install the package itself in development mode (fast, as dependencies are already installed)
RUN pip install --no-cache-dir -e .

