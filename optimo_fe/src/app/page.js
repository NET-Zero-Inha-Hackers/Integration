"use client";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState, useMemo } from "react";
import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export default function Home() {
  const router = useRouter();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  // 빛나는 점들의 위치를 컴포넌트 초기화 시에만 계산
  const glowingDots = useMemo(() => {
    return [...Array(30)].map((_, i) => ({
      id: i,
      top: `${Math.random() * 95}%`,
      left: `${Math.random() * 95}%`,
      animationDelay: `${Math.random() * 2}s`
    }));
  }, []);

  const handleEmailChange = (e) => {
    setEmail(e.target.value);
    // 이메일 입력 시 에러 메시지 초기화
    if (error) {
      setError("");
    }
  };

  const handleLogin = async () => {
    if (!email.trim()) {
      setError("이메일을 입력해주세요.");
      return;
    }

    setIsLoading(true);
    setError("");

    try {
      const userInfo = await api.getUserInfo(email);
      console.log("로그인 성공:", userInfo);
      
      // AuthContext에 사용자 정보 저장
      login(userInfo);
      
      console.log("✅ 세션에 저장된 사용자 정보:", {
        id: userInfo.id,
        email: userInfo.email,
        name: userInfo.name,
        profileImage: userInfo.profileImage,
        provider: userInfo.provider,
        totalUseElecEstimate: userInfo.totalUseElecEstimate,
        totalLlmElecEstimate: userInfo.totalLlmElecEstimate
      });
      
      router.push('/chat');
    } catch (error) {
      console.error("로그인 실패:", error);
      setError("로그인에 실패했습니다. 이메일을 확인해주세요.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      handleLogin();
    }
  };

  return (
    <>
      <div className="flex h-screen bg-black overflow-y-hidden">
        {/* 왼쪽: Optimo 인트로 */}
        <div className="relative flex flex-col justify-center items-center w-1/2 h-full bg-black overflow-hidden animate-slideup">
          {/* 빛나는 점 효과 */}
          <div className="absolute inset-0 z-0">
            {glowingDots.map((dot) => (
              <div
                key={dot.id}
                className="absolute rounded-full bg-lime-400 opacity-100 blur-[1px] animate-glow"
                style={{
                  width: '6px',
                  height: '6px',
                  top: dot.top,
                  left: dot.left,
                  animationDelay: dot.animationDelay
                }}
              />
            ))}
          </div>
          {/* 로고 */}
          <div className="z-10 mb-6">
            <Image src="/icon/optimo.png" alt="Optimo Logo" width={80} height={80} className="mx-auto" />
          </div>
          {/* 타이틀 */}
          <h1 className="z-10 text-5xl font-bold text-lime-400 mb-8">Optimo</h1>
          {/* 설명 문구 */}
          <div className="z-10 text-center text-2xl font-semibold">
            <p className="text-white">
              필요할 때만 <span className="text-lime-400">AI의 힘을</span><br />
              <span className="text-lime-400">지구를 생각</span>하는 스마트한 선택
            </p>
          </div>
        </div>

        {/* 오른쪽: 로그인 폼 */}
        <div className="flex flex-col justify-center items-center w-1/2 h-full bg-black">
          <div className="w-80">
            <h2 className="text-white text-2xl font-semibold mb-6 text-center">로그인</h2>
            
            {/* 이메일 입력 */}
            <input
              type="email"
              placeholder="이메일 주소"
              value={email}
              onChange={handleEmailChange}
              onKeyPress={handleKeyPress}
              className="w-full mb-4 px-4 py-3 rounded-full outline-none text-black text-xs font-semibold h-12"
              disabled={isLoading}
            />
            
            {/* 에러 메시지 */}
            {error && (
              <div className="mb-4 text-red-400 text-sm text-center">
                {error}
              </div>
            )}
            
            {/* 로그인 버튼 */}
            <button
              onClick={handleLogin}
              disabled={isLoading}
              className="w-full py-3 rounded-full bg-lime-400 text-white text-xl font-bold hover:bg-lime-500 transition-colors h-12 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "로그인 중..." : "제출"}
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
