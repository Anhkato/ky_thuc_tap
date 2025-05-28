import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import apiClient from './api';

async function initializeApp() {
  try {
    await apiClient.get('/sanctum/csrf-cookie');
  } catch (error) {
    console.error('Failed to fetch CSRF cookie:', error);
  } finally {
    const app = createApp(App);
    app.use(router);
    app.mount('#app');
  }
}

initializeApp();