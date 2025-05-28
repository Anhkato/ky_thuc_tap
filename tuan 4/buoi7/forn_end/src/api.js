
import axios from 'axios';

function getCookie(name) {
  console.log('[getCookie] Đang tìm cookie. Chuỗi document.cookie:', document.cookie);
  const value = `; ${document.cookie}`;
  const parts = value.split(`; ${name}=`);
  if (parts.length === 2) {
    const cookieValue = parts.pop().split(';').shift();
    console.log(`[getCookie] Giá trị thô cho cookie "${name}":`, cookieValue);
    try {
      const decodedValue = decodeURIComponent(cookieValue);
      console.log(`[getCookie] Giá trị đã decode cho cookie "${name}":`, decodedValue);
      return decodedValue;
    } catch (e) {
      console.error(`[getCookie] Lỗi khi decode cookie "${name}":`, e, 'Giá trị thô là:', cookieValue);
      return cookieValue; 
    }
  }
  console.log(`[getCookie] Không tìm thấy cookie "${name}".`);
  return null;
}

const apiClient = axios.create({
  baseURL: 'http://localhost:8000',
  withCredentials: true,
});

apiClient.interceptors.request.use(
  (config) => {
    console.log('[Axios Interceptor] Đang chạy cho request đến:', config.url, 'Phương thức:', config.method);
    const methodsToAttachToken = ['post', 'put', 'delete', 'patch'];
    if (methodsToAttachToken.includes(config.method.toLowerCase())) {
      console.log('[Axios Interceptor] Đang thử đính kèm XSRF token.');
      const token = getCookie('XSRF-TOKEN');
      if (token) {
        config.headers['X-XSRF-TOKEN'] = token;
        console.log('[Axios Interceptor] Header X-XSRF-TOKEN ĐÃ ĐƯỢC SET thành:', token);
      } else {
        console.warn('[Axios Interceptor] Cookie XSRF-TOKEN không được tìm thấy bởi getCookie.');
      }
    }

    try {
        console.log('[Axios Interceptor] Các header cuối cùng của request:', JSON.parse(JSON.stringify(config.headers)));
    } catch (e) {
        console.log('[Axios Interceptor] Các header cuối cùng của request (không thể JSON stringify):', config.headers);
    }
    return config;
  },
  (error) => {
    console.error('[Axios Interceptor] Lỗi trong request interceptor:', error);
    return Promise.reject(error);
  }
);

export default apiClient;