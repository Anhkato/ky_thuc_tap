import { createRouter, createWebHistory } from 'vue-router';
import ProductList from '../components/views/ProductList.vue';
import ProductDetail from '../components/views/ProductDetail.vue';
import ProductForm from '../components/views/ProductForm.vue';
import Login from '../components/views/Login.vue';
import apiClient from '../api.js';

async function checkAuth() {
  try {
    const response = await apiClient.get('/api/user');
    return !!response.data;
  } catch (error) {
    return false;
  }
}

const routes = [
  {
    path: '/login',
    name: 'Login',
    component: Login,
    beforeEnter: async (to, from, next) => {
      if (await checkAuth()) {
        next({ name: 'ProductList' });
      } else {
        next();
      }
    },
  },
  {
    path: '/',
    name: 'ProductList',
    component: ProductList,
    meta: { requiresAuth: true },
  },
  {
    path: '/products/create',
    name: 'ProductCreate',
    component: ProductForm,
    meta: { requiresAuth: true },
    props: { isEditMode: false }
  },
  {
    path: '/products/:id',
    name: 'ProductDetail',
    component: ProductDetail,
    meta: { requiresAuth: true },
    props: true,
  },
  {
    path: '/products/:id/edit',
    name: 'ProductEdit',
    component: ProductForm,
    meta: { requiresAuth: true },
    props: route => ({ isEditMode: true, productId: String(route.params.id) })
  },
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
});

router.beforeEach(async (to, from, next) => {
  const requiresAuth = to.matched.some(record => record.meta.requiresAuth);
  const isAuthenticated = await checkAuth();

  if (requiresAuth && !isAuthenticated) {
    next({ name: 'Login' });
  } else if (to.name === 'Login' && isAuthenticated) {
    next({ name: 'ProductList' });
  }
  else {
    next();
  }
});

export default router;