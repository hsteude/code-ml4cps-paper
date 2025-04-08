# Stage 1: Build dependencies
FROM python:3.11 as builder

WORKDIR /app

# Install UV
RUN pip install --no-cache-dir uv

# Copy only the pyproject.toml file
COPY pyproject.toml ./
# Create empty directories for the packages defined in pyproject.toml
RUN mkdir -p container_component_src pipeline
# Copy the README.md file that is referenced in pyproject.toml
COPY README.md ./

# Install dependencies only using UV
RUN uv pip install --system .

# Stage 2: Final image
FROM python:3.11

WORKDIR /app

# Install UV in the final image
RUN pip install --no-cache-dir uv

# Copy installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Now copy the actual code
COPY ./container_component_src /app/container_component_src
COPY ./pipeline /app/pipeline
COPY ./config.toml /app/config.toml
COPY ./README.md /app/
COPY pyproject.toml ./

# Install the package itself in development mode using UV
RUN uv pip install --system -e .
