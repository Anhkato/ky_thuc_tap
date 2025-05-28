<template>
  <div>
    <h2>{{ formTitle }}</h2>
    <form @submit.prevent="submitForm">
      <div>
        <label for="name">Name:</label>
        <input type="text" id="name" v-model="form.name" required>
      </div>
      <div>
        <label for="description">Description:</label>
        <textarea id="description" v-model="form.description"></textarea>
      </div>
      <div>
        <label for="price">Price:</label>
        <input type="number" id="price" v-model="form.price" required min="0" step="0.01">
      </div>
      <div>
        <label for="image_file">Image:</label>
        <input type="file" id="image_file" @change="handleImageUpload">
      </div>
      <div v-if="imagePreviewUrl">
        <p>Image Preview:</p>
        <img :src="imagePreviewUrl" alt="Image Preview" class="product-image-preview"/>
      </div>
       <div v-if="form.image_url && !imagePreviewUrl && isEditMode">
        <p>Current Image:</p>
        <img :src="form.image_url" alt="Current Image" class="product-image-preview"/>
      </div>

      <button type="submit" :disabled="submitting">
        {{ submitting ? 'Saving...' : 'Save Product' }}
      </button>
      <p v-if="errorMessage" class="error-message">{{ errorMessage }}</p>
      <div v-if="validationErrors">
        <ul class="error-message">
          <li v-for="(errors, field) in validationErrors" :key="field">
            {{ field }}: {{ errors.join(', ') }}
          </li>
        </ul>
      </div>
    </form>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import apiClient from '../../api';

const props = defineProps({
  isEditMode: {
    type: Boolean,
    default: false,
  },
  productId: {
    type: String,
    default: null,
  }
});

const router = useRouter();
const route = useRoute();

const form = ref({
  name: '',
  description: '',
  price: 0,
  image_file: null,
  image_url: null, 
});
const imagePreviewUrl = ref(null);
const submitting = ref(false);
const errorMessage = ref('');
const validationErrors = ref(null);


const formTitle = computed(() => props.isEditMode ? 'Edit Product' : 'Create New Product');

const handleImageUpload = (event) => {
  const file = event.target.files[0];
  if (file) {
    form.value.image_file = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreviewUrl.value = e.target.result;
    };
    reader.readAsDataURL(file);
  } else {
    form.value.image_file = null;
    imagePreviewUrl.value = null;
  }
};

const fetchProductForEdit = async () => {
  if (!props.isEditMode || !props.productId) return;
  submitting.value = true;
  try {
    const response = await apiClient.get(`/api/products/${props.productId}`);
    const productData = response.data;
    form.value.name = productData.name;
    form.value.description = productData.description;
    form.value.price = productData.price;
    form.value.image_url = productData.image_url; 
  } catch (error) {
    console.error('Failed to fetch product for editing:', error);
    errorMessage.value = 'Could not load product data for editing.';
  } finally {
    submitting.value = false;
  }
};

const submitForm = async () => {
  submitting.value = true;
  errorMessage.value = '';
  validationErrors.value = null;

  const formData = new FormData();
  formData.append('name', form.value.name);
  formData.append('description', form.value.description || '');
  formData.append('price', form.value.price);
  if (form.value.image_file) {
    formData.append('image_file', form.value.image_file);
  }

  try {
    if (props.isEditMode) {
      formData.append('_method', 'PUT'); // Important for Laravel to recognize PUT
      await apiClient.post(`/api/products/${props.productId}`, formData, {
         headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    } else {
      await apiClient.post('/api/products', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
    }
    router.push({ name: 'ProductList' });
  } catch (error) {
    console.error('Error submitting form:', error);
    if (error.response && error.response.status === 422) {
      validationErrors.value = error.response.data.errors;
      errorMessage.value = 'Please correct the validation errors.';
    } else if (error.response && error.response.data && error.response.data.message) {
      errorMessage.value = error.response.data.message;
    } else {
      errorMessage.value = `An error occurred: ${error.message || 'Unknown error'}`;
    }
  } finally {
    submitting.value = false;
  }
};

onMounted(() => {
  if (props.isEditMode) {
    fetchProductForEdit();
  }
});

watch(() => props.productId, (newId, oldId) => {
    if (props.isEditMode && newId && newId !== oldId) {
        form.value = { name: '', description: '', price: 0, image_file: null, image_url: null };
        imagePreviewUrl.value = null;
        fetchProductForEdit();
    } else if (!props.isEditMode) {
         form.value = { name: '', description: '', price: 0, image_file: null, image_url: null }; 
         imagePreviewUrl.value = null;
    }
});

</script>