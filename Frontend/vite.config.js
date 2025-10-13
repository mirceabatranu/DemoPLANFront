import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Explicitly set the project root
  root: '.', 
  build: {
    // Output directory for the build files
    outDir: 'dist',
  },
});