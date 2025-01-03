# WorldForge/src/frontend/Dockerfile

FROM node:22.12.0-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

RUN npm install

# Build the project
RUN npm run build

# Install a simple HTTP server to serve static files
RUN npm install -g http-server

# Expose port
EXPOSE 8080

# Serve the built static files
CMD ["http-server", "dist", "-p", "8080"]
