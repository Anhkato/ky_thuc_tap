<template>
  <div class="login-form">
    <h2>Login</h2>
    <form @submit.prevent="handleLogin">
      <div>
        <label for="email">Email:</label>
        <input type="email" id="email" v-model="email" required>
      </div>
      <div>
        <label for="password">Password:</label>
        <input type="password" id="password" v-model="password" required>
      </div>
      <button type="submit" :disabled="loading">
        {{ loading ? 'Logging in...' : 'Login' }}
      </button>
      <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
    </form>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import apiClient from '../../api';

const email = ref('');
const password = ref('');
const errorMessage = ref('');
const loading = ref(false);
const router = useRouter();

const handleLogin = async () => {
  errorMessage.value = '';
  loading.value = true;
  try {
    await apiClient.get('/sanctum/csrf-cookie');
    await apiClient.post('/login', {
      email: email.value,
      password: password.value,
    });
  
    window.location.href = '/';
  } catch (error) {
    if (error.response && error.response.status === 422) {
      errorMessage.value = 'Invalid credentials or validation error. Please check your input.';
      if (error.response.data && error.response.data.errors) {
        console.log(error.response.data.errors)
      }
    } else {
      errorMessage.value = 'Login failed. Please try again later.';
    }
    console.error('Login error:', error);
  } finally {
    loading.value = false;
  }
};
</script>