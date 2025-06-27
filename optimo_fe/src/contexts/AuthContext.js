"use client";
import { createContext, useContext, useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const router = useRouter();
  useEffect(() => {
    // 페이지 로드 시 localStorage에서 사용자 정보 복원
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        setUser(userData);
        console.log('✅ 저장된 사용자 정보 복원:', userData);
      } catch (error) {
        console.error('❌ 저장된 사용자 정보 파싱 실패:', error);
        localStorage.removeItem('user');
      }
    }
    setIsLoading(false);
  }, []);

  const login = (userInfo) => {
    // 사용자 정보 구조:
    // {
    //   id: 12345,
    //   email: "user@example.com",
    //   name: "홍길동",
    //   profileImage: "https://example.com/profile.jpg",
    //   provider: "EMAIL",
    //   totalUseElecEstimate: 1500,
    //   totalLlmElecEstimate: 500
    // }
    setUser(userInfo);
    localStorage.setItem('user', JSON.stringify(userInfo));
    console.log('✅ 사용자 정보 저장:', userInfo);
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('lastChattingId');
    console.log('✅ 사용자 정보 삭제');
    document.cookie = "jwtToken=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    router.push('/');

  };

  const updateUser = (updates) => {
    if (user) {
      const updatedUser = { ...user, ...updates };
      setUser(updatedUser);
      localStorage.setItem('user', JSON.stringify(updatedUser));
      console.log('✅ 사용자 정보 업데이트:', updatedUser);
    }
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      login, 
      logout, 
      updateUser,
      isLoading,
      isAuthenticated: !!user 
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext); 