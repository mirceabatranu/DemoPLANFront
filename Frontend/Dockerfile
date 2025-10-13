# Stage 1: Build the React application
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve the application using a lightweight Nginx server
FROM nginx:1.25-alpine AS final
# Copy the built files from the 'build' stage
COPY --from=build /app/dist /usr/share/nginx/html
# Copy our custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf
# Expose port 8080 to match Cloud Run's default
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]