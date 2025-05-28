<template>
  <div>
    <div v-if="loading">Loading product details...</div>
    <div v-if="error" class="error-message">{{ error }}</div>
    <div v-if="product">
      <h2>{{ product.name }}</h2>
      <p><strong>Description:</strong> {{ product.description || 'N/A' }}</p>
      <p><strong>Price:</strong> ${{ product.price }}</p>
<img v-if="product.image_url" :src="`http://127.0.0.1:8000/${product.image_url.startsWith('/') ? product.image_url.substring(1) : product.image_url}`" :alt="product.name" class="product-image-preview"/>      <router-link :to="{ name: 'ProductList' }">Back to List</router-link>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { useRoute } from 'vue-router';
import apiClient from '../../api';

const route = useRoute();
const product = ref(null);
const loading = ref(true);
const error = ref('');

const fetchProduct = async () => {
  loading.value = true;
  error.value = '';
  try {
    const response = await apiClient.get(`/api/products/${route.params.id}`);
    product.value = response.data;
  } catch (err) {
    console.error('Error fetching product details:', err);
    error.value = 'Failed to load product details.';
  } finally {
    loading.value = false;
  }
};

onMounted(fetchProduct);
</script>