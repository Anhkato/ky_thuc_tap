<template>
  <div class="product-list-container">
    <h2 class="list-title">Danh Sách Sản Phẩm</h2>

    <div v-if="loading" class="status-message loading-message">
      <p>Đang tải sản phẩm, vui lòng chờ...</p>
    </div>

    <div v-if="error" class="status-message error-message alert alert-danger">
      <p><strong>Đã xảy ra lỗi:</strong> {{ error }}</p>
    </div>

    <div v-if="!loading && !error && products.length" class="products-display-area">
      <ul class="product-list-enhanced">
        <li v-for="product in products" :key="product.id" class="product-list-item">
          <div class="product-info">
            <router-link :to="{ name: 'ProductDetail', params: { id: product.id } }" class="product-name-link">
              {{ product.name }}
            </router-link>
            <span class="product-price-tag">- {{ getDisplayPrice(product.price) }}</span>
          </div>
          <div class="product-actions-group">
            <router-link :to="{ name: 'ProductEdit', params: { id: product.id } }" class="action-link edit-link">
              Chỉnh sửa
            </router-link>
            <button @click="deleteProduct(product.id)" class="action-button delete-button">
              Xóa
            </button>
          </div>
        </li>
      </ul>
      <div class="pagination-controls" v-if="pagination.last_page > 1">
        <button
          :disabled="pagination.current_page === 1"
          @click="fetchProducts(pagination.current_page - 1)"
        >Trang trước</button>
        <span>Trang {{ pagination.current_page }} / {{ pagination.last_page }}</span>
        <button
          :disabled="pagination.current_page === pagination.last_page"
          @click="fetchProducts(pagination.current_page + 1)"
        >Trang sau</button>
      </div>
    </div>

    <div v-if="!loading && !error && !products.length" class="status-message no-products-message">
      <p>Hiện tại không có sản phẩm nào.</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import apiClient from '../../api'

const products = ref([])
const loading = ref(true)
const error = ref('')
const pagination = ref({
  current_page: 1,
  last_page: 1,
  per_page: 10,
  total: 0
})

const getDisplayPrice = (price) => {
  const numPrice = parseFloat(price)
  if (!isNaN(numPrice) && isFinite(numPrice)) {
    return `$${numPrice.toFixed(2)}`
  }
  return 'Giá: Liên hệ'
}

const fetchProducts = async (page = 1) => {
  loading.value = true
  error.value = ''
  try {
    const response = await apiClient.get(`/api/products?page=${page}`)
    products.value = response.data.data
    pagination.value = {
      current_page: response.data.current_page,
      last_page: response.data.last_page,
      per_page: response.data.per_page,
      total: response.data.total
    }
  } catch (err) {
    if (err.response && err.response.data && err.response.data.message) {
      error.value = `Failed to load products: ${err.response.data.message}`
    } else if (err.message) {
      error.value = `Failed to load products: ${err.message}`
    } else {
      error.value = 'Failed to load products. An unknown error occurred.'
    }
  } finally {
    loading.value = false
  }
}

const deleteProduct = async (id) => {
  if (confirm('Bạn có chắc chắn muốn xóa sản phẩm này không?')) {
    try {
      await apiClient.delete(`/api/products/${id}`)
      fetchProducts(pagination.value.current_page)
    } catch (err) {
      let deleteErrorMessage = 'Failed to delete product.'
      if (err.response && err.response.data && err.response.data.message) {
        deleteErrorMessage = `Failed to delete product: ${err.response.data.message}`
      } else if (err.message) {
        deleteErrorMessage = `Failed to delete product: ${err.message}`
      }
      alert(deleteErrorMessage)
    }
  }
}

onMounted(() => fetchProducts())
</script>

<style scoped>
.product-list-container {
  max-width: 960px;
  margin: 25px auto;
  padding: 20px 25px;
  background-color: #f9f9f9;
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.list-title {
  text-align: center;
  color: #333;
  margin-bottom: 25px;
  font-size: 1.8em;
  font-weight: 600;
}

.status-message {
  text-align: center;
  padding: 18px;
  margin-bottom: 20px;
  border-radius: 6px;
  font-size: 1.05em;
}

.loading-message {
  color: #555;
}

.error-message.alert-danger {
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}

.no-products-message {
  color: #777;
}

.product-list-enhanced {
  list-style-type: none;
  padding: 0;
}

.product-list-item {
  background-color: #ffffff;
  border: 1px solid #e7e7e7;
  border-radius: 6px;
  padding: 15px 20px;
  margin-bottom: 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: box-shadow 0.2s ease;
}

.product-list-item:hover {
  box-shadow: 0 3px 8px rgba(0,0,0,0.07);
}

.product-info {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.product-name-link {
  font-size: 1.15em;
  color: #007bff;
  text-decoration: none;
  font-weight: 500;
}
.product-name-link:hover {
  text-decoration: underline;
  color: #0056b3;
}

.product-price-tag {
  font-size: 1em;
  color: #28a745;
  font-weight: bold;
}

.product-actions-group {
  display: flex;
  align-items: center;
  gap: 12px;
}

.action-link {
  color: #6c757d;
  text-decoration: none;
  font-size: 0.95em;
  padding: 5px 8px;
  border-radius: 4px;
  transition: background-color 0.2s ease, color 0.2s ease;
}
.action-link.edit-link:hover {
  background-color: #e9ecef;
  color: #007bff;
}

.action-button {
  padding: 6px 12px;
  font-size: 0.9em;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.action-button.delete-button {
  background-color: #f8d7da;
  color: #721c24;
  border: 1px solid #f5c6cb;
}
.action-button.delete-button:hover {
  background-color: #f1b0b7;
}

.pagination-controls {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 16px;
  margin-top: 18px;
}

.pagination-controls button {
  padding: 6px 14px;
  border: none;
  border-radius: 4px;
  background: #e9ecef;
  color: #333;
  cursor: pointer;
  font-size: 1em;
  transition: background 0.2s;
}
.pagination-controls button:disabled {
  background: #f1f1f1;
  color: #aaa;
  cursor: not-allowed;
}
.pagination-controls span {
  font-size: 1.1em;
  font-weight: 500;
}

@media (max-width: 600px) {
  .product-list-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  .product-actions-group {
    width: 100%;
    justify-content: flex-start;
  }
}
</style>