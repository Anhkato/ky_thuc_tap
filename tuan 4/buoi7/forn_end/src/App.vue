<template>
  <div id="app-container">
    <header v-if="isLoggedIn">
      <nav>
        <router-link to="/">Products</router-link>
        <router-link to="/products/create">Create Product</router-link>
        <button @click="performLogout">Logout</button>
      </nav>
    </header>
    <main>
      <router-view />
    </main>
  </div>
</template>

<script setup>
import { ref, provide, readonly, onMounted, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import apiClient from './api';

const router = useRouter();
const route = useRoute();
const isLoggedIn = ref(false);
const currentUser = ref(null);

provide('currentUser', readonly(currentUser));
provide('isLoggedIn', readonly(isLoggedIn));


const checkLoginStatus = async () => {
  try {
    const response = await apiClient.get('/api/user');
    currentUser.value = response.data;
    isLoggedIn.value = true;
  } catch {
    currentUser.value = null;
    isLoggedIn.value = false;
  }
};

const performLogout = async () => {
  try {
    await apiClient.post('/logout');
  } catch (error) {
    console.error('Logout API call failed:', error);
  } finally {
    currentUser.value = null;
    isLoggedIn.value = false;
    router.push({ name: 'Login' });
  }
};

onMounted(checkLoginStatus);
watch(() => route.name, async (newName) => {
  if (newName !== 'Login') {
    await checkLoginStatus();
  }
});

</script>

<style>
#app-container { padding: 20px; }
header { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 10px;}
nav a, nav button { margin-right: 15px; text-decoration: none; color: #007bff; }
nav button { background: none; border: none; cursor: pointer; padding: 0; font-size: inherit; }
nav a.router-link-exact-active { font-weight: bold; }
main { margin-top: 20px; }
button { padding: 8px 15px; margin: 5px; cursor: pointer; }
div { margin-bottom: 10px; }
label { display: block; margin-bottom: 5px; }
input[type="text"], input[type="email"], input[type="password"], input[type="number"], textarea {
  width: 100%;
  padding: 8px;
  margin-bottom: 10px;
  box-sizing: border-box;
}
form button { background-color: #28a745; color: white; border: none; }
ul { list-style: none; padding: 0; }
li { padding: 10px; border-bottom: 1px solid #eee; }
li button { margin-left: 10px; background-color: #dc3545; color:white; border: none;}
li a { margin-right: 10px;}
.error-message { color: red; margin-top: 10px; }
.product-image-preview { max-width: 200px; max-height: 200px; margin-top: 10px; }
</style>