# --- Stage 1: Build React App ---
FROM node:20-alpine AS build

WORKDIR /app

# Copy and install dependencies
COPY frontend/package*.json ./frontend/
WORKDIR /app/frontend
RUN npm install

# Copy rest of frontend
COPY frontend/ .

# Build production version
RUN npm run build

# --- Stage 2: Serve the app ---
FROM node:20-alpine

WORKDIR /app

# Install a lightweight HTTP server for static content
RUN npm install -g serve

# Copy build files from Stage 1
COPY --from=build /app/frontend/dist ./dist

# Expose port
EXPOSE 5173

# Serve the app
CMD ["serve", "-s", "dist", "-l", "5173"]
