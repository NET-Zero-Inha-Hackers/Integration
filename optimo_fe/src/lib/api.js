// API 클라이언트 - axios와 동일한 문법, 자동 토큰 갱신
class ApiClient {
    constructor() {
      this.baseURL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';
      this.isRefreshing = false;
      this.failedQueue = [];
    }
  
    // 쿠키에서 JWT 토큰 가져오기
    async getToken() {
      if (typeof window !== 'undefined') {
        const cookies = document.cookie.split(';');
        const tokenCookie = cookies.find(cookie => cookie.trim().startsWith('jwtToken='));
        return tokenCookie ? tokenCookie.split('=')[1] : null;
      }
      return null;
    }
  
    // 토큰을 쿠키에 저장
    setToken(token) {
      if (typeof window !== 'undefined') {
        document.cookie = `jwtToken=${token}; path=/; max-age=${7 * 24 * 60 * 60}; secure; samesite=lax`;
      }
    }
  
    // 토큰을 쿠키에서 제거
    removeToken() {
      if (typeof window !== 'undefined') {
        document.cookie = 'jwtToken=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';
      }
    }
  
    // 기본 헤더 설정
    async getHeaders(customHeaders = {}) {
      const token = await this.getToken();
      const headers = {
        'Content-Type': 'application/json',
        ...customHeaders,
      };
      
      if (token) {
        headers['Authorization'] = `Bearer ${token}`;
      }
      
      return headers;
    }
  
    // 토큰 갱신
    async refreshToken() {
      try {
        console.log('🔄 토큰 갱신 시도...');
        
        const response = await fetch(`${this.baseURL}/auth/refresh`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          credentials: 'include', // 쿠키 포함
        });
  
        if (response.ok) {
          const data = await response.json();
          this.setToken(data.token);
          console.log('✅ 토큰 갱신 성공');
          return data.token;
        } else {
          throw new Error('토큰 갱신 실패');
        }
      } catch (error) {
        console.error('❌ 토큰 갱신 실패:', error);
        this.removeToken();
        window.location.href = '/auth/login';
        throw error;
      }
    }
  
    // 실패한 요청들을 재실행
    processQueue(error, token = null) {
      this.failedQueue.forEach(({ resolve, reject }) => {
        if (error) {
          reject(error);
        } else {
          resolve(token);
        }
      });
      
      this.failedQueue = [];
    }
  
    // axios와 동일한 request 메서드
    async request(config) {
      const {
        method = 'GET',
        url,
        data,
        params,
        headers = {},
        ...otherConfig
      } = config;
  
      const makeRequest = async (token = null) => {
        const requestHeaders = await this.getHeaders(headers);
        const requestUrl = new URL(`${this.baseURL}${url}`);
        
        // 쿼리 파라미터 추가
        if (params) {
          Object.keys(params).forEach(key => {
            requestUrl.searchParams.append(key, params[key]);
          });
        }
  
        const fetchConfig = {
          method: method.toUpperCase(),
          headers: requestHeaders,
          ...otherConfig,
        };
  
        // body가 있는 경우에만 추가
        if (data && method.toUpperCase() !== 'GET') {
          fetchConfig.body = JSON.stringify(data);
        }
  
        const response = await fetch(requestUrl.toString(), fetchConfig);
        
        // 토큰 만료 체크 (401 에러)
        if (response.status === 401 && !url.includes('/auth/refresh')) {
          if (!this.isRefreshing) {
            this.isRefreshing = true;
            
            try {
              const newToken = await this.refreshToken();
              this.processQueue(null, newToken);
              this.isRefreshing = false;
              
              // 새로운 토큰으로 재요청
              return makeRequest(newToken);
            } catch (error) {
              this.processQueue(error, null);
              this.isRefreshing = false;
              throw error;
            }
          } else {
            // 이미 토큰 갱신 중이면 큐에 추가
            return new Promise((resolve, reject) => {
              this.failedQueue.push({ resolve, reject });
            }).then(token => makeRequest(token))
              .catch(error => {
                throw error;
              });
          }
        }
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
      };
  
      return makeRequest();
    }
  
    // axios와 동일한 메서드들
    async get(url, config = {}) {
      return this.request({
        method: 'GET',
        url,
        ...config,
      });
    }
  
    async post(url, data, config = {}) {
      return this.request({
        method: 'POST',
        url,
        data,
        ...config,
      });
    }
  
    async put(url, data, config = {}) {
      return this.request({
        method: 'PUT',
        url,
        data,
        ...config,
      });
    }
  
    async patch(url, data, config = {}) {
      return this.request({
        method: 'PATCH',
        url,
        data,
        ...config,
      });
    }
  
    async delete(url, config = {}) {
      return this.request({
        method: 'DELETE',
        url,
        ...config,
      });
    }
  
    // 파일 업로드 (FormData)
    async upload(url, file, additionalData = {}, config = {}) {
      const makeUploadRequest = async (token = null) => {
        const formData = new FormData();
        formData.append('file', file);
        
        Object.keys(additionalData).forEach(key => {
          formData.append(key, additionalData[key]);
        });
        
        const headers = { ...config.headers };
        
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        } else {
          const currentToken = await this.getToken();
          if (currentToken) {
            headers['Authorization'] = `Bearer ${currentToken}`;
          }
        }
        
        const response = await fetch(`${this.baseURL}${url}`, {
          method: 'POST',
          headers,
          body: formData,
          ...config,
        });
        
        // 토큰 만료 체크
        if (response.status === 401 && !url.includes('/auth/refresh')) {
          if (!this.isRefreshing) {
            this.isRefreshing = true;
            
            try {
              const newToken = await this.refreshToken();
              this.processQueue(null, newToken);
              this.isRefreshing = false;
              
              return makeUploadRequest(newToken);
            } catch (error) {
              this.processQueue(error, null);
              this.isRefreshing = false;
              throw error;
            }
          } else {
            return new Promise((resolve, reject) => {
              this.failedQueue.push({ resolve, reject });
            }).then(token => makeUploadRequest(token))
              .catch(error => {
                throw error;
              });
          }
        }
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
      };
  
      return makeUploadRequest();
    }
  
    // axios와 동일한 인터셉터 기능
    interceptors = {
      request: {
        use: (fulfilled, rejected) => {
          this.requestInterceptor = { fulfilled, rejected };
        },
      },
      response: {
        use: (fulfilled, rejected) => {
          this.responseInterceptor = { fulfilled, rejected };
        },
      },
    };
  
    // 인터셉터 적용
    async applyInterceptors(config, type) {
      if (type === 'request' && this.requestInterceptor) {
        try {
          return this.requestInterceptor.fulfilled(config);
        } catch (error) {
          if (this.requestInterceptor.rejected) {
            return this.requestInterceptor.rejected(error);
          }
          throw error;
        }
      }
      return config;
    }
  
    // 사용자 정보 가져오기 (토큰을 헤더에서 받아서 쿠키에 저장)
    async getUserInfo(email) {
      try {
        const response = await fetch(`${this.baseURL}/user?email=${email}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });

        
  
        if (response.ok) {

            const userData = await response.json();
            console.log('✅ 사용자 정보:', userData);

            const authToken = userData.token;

          
          if (authToken) {
            this.setToken(authToken);
            console.log('✅ 토큰을 쿠키에 저장했습니다');
          } else {
            console.log('⚠️ Authorization 헤더가 없거나 Bearer 형식이 아닙니다');
          }
          
          return userData;
        } else {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
      } catch (error) {
        console.error('❌ 사용자 정보 가져오기 실패:', error);
        throw error;
      }
    }
  }
  
  // 싱글톤 인스턴스 생성
  const apiClient = new ApiClient();
  
  // axios와 동일한 사용법을 위한 함수들
  export const api = {
    // 기본 request 메서드
    request: (config) => apiClient.request(config),
    
    // HTTP 메서드별 함수들
    get: (url, config) => apiClient.get(url, config),
    post: (url, data, config) => apiClient.post(url, data, config),
    put: (url, data, config) => apiClient.put(url, data, config),
    patch: (url, data, config) => apiClient.patch(url, data, config),
    delete: (url, config) => apiClient.delete(url, config),
    
    // 파일 업로드
    upload: (url, file, additionalData, config) => apiClient.upload(url, file, additionalData, config),
    
    // 사용자 정보 가져오기
    getUserInfo: (email) => apiClient.getUserInfo(email),
    
    // 인터셉터
    interceptors: apiClient.interceptors,
  };
  
  export default apiClient;
  