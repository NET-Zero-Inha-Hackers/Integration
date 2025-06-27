# Optimo Frontend

<img width="1455" alt="image" src="https://github.com/user-attachments/assets/9a8bd74e-808d-47d3-be89-0fbb5ef32ef5" />


> **지구를 생각하는 AI 채팅 플랫폼** 🌱

Optimo는 AI 사용 시 발생하는 전력 소비와 탄소배출을 고려하여, **Net Zero** 달성을 위한 스마트한 AI 사용을 제안하는 플랫폼입니다. 필요할 때 적합한 AI의 힘을 사용하여 환경에 미치는 영향을 최소화합니다.

## 🌍 Net Zero 미션

- **탄소배출 추적**: AI 사용으로 인한 탄소발자국 실시간 모니터링
- **친환경 사용 가이드**: 탄소중립을 위한 AI 사용 최적화 제안
- **Net Zero 목표 설정**: 개인별 탄소중립 달성 목표 및 진행률 추적

## 🚀 기술 스택

- **Framework**: Next.js 14 (App Router)
- **Language**: JavaScript (ES6+)
- **Styling**: Tailwind CSS 3.4.1
- **State Management**: React Context API
- **Markdown Rendering**: React Markdown + KaTeX (수학 공식 지원)
- **UI Components**: Custom Components
- **Authentication**: JWT Token (쿠키 기반)
- **Real-time Communication**: WebSocket

## ⚡ 핵심 기능

### 🔐 **간편 로그인 시스템**
- **이메일만으로 로그인**: 복잡한 비밀번호 없이 이메일 주소만 입력
- **자동 세션 관리**: 브라우저를 닫아도 로그인 상태 유지
- **보안 토큰**: JWT 기반 안전한 인증

### 💬 **스마트 AI 채팅**
- **실시간 대화**: AI와 즉시 소통 가능
- **마크다운 지원**: 코드 블록, 링크, 이미지 등 풍부한 텍스트 표현
- **수학 공식 렌더링**: KaTeX를 통한 수학 공식 표시
- **채팅 히스토리**: 이전 대화 내용 저장 및 관리
- **새 채팅 시작**: 언제든지 새로운 대화 시작 가능

### 📊 **탄소배출 추적 시스템**
- **실시간 탄소발자국 계산**: AI 사용 시 발생하는 CO2 배출량 추적
- **전력 소비량 모니터링**: AI 사용으로 인한 전력 소비 실시간 추적
- **환경 영향 환산**: 탄소배출량을 나무 심기, 자동차 주행거리 등으로 환산
- **Net Zero 진행률**: 개인별 탄소중립 달성 목표 및 진행 상황 표시
- **친환경 사용 패턴 분석**: 효율적인 AI 사용으로 탄소배출 최소화 가이드

### 🎨 **직관적인 사용자 인터페이스**
- **다크 테마**: 눈의 피로도를 줄이는 어두운 배경
- **사이드바 네비게이션**: 한 번의 클릭으로 페이지 이동
- **반응형 디자인**: 모바일, 태블릿, 데스크톱 모든 기기 지원
- **부드러운 애니메이션**: 사용자 경험을 향상시키는 인터랙션

### 👤 **개인화된 프로필 관리**
- **사용자 정보 관리**: 이름, 이메일, 프로필 이미지 설정
- **탄소중립 목표 설정**: 개인별 Net Zero 달성 목표 설정
- **환경 친화 점수**: AI 사용 효율성 및 탄소배출 감소 점수 제공
- **환경 기여도 시각화**: 개인이 절약한 탄소량을 시각적으로 표현

## 📁 프로젝트 구조

```
src/
├── app/                    # Next.js App Router
│   ├── (layout)/          # 레이아웃 그룹
│   │   ├── chat/          # 채팅 페이지
│   │   ├── overview/      # 개요 페이지
│   │   └── profile/       # 프로필 페이지
│   ├── auth/              # 인증 관련
│   ├── globals.css        # 전역 스타일
│   ├── layout.js          # 루트 레이아웃
│   └── page.js            # 홈페이지 (로그인)
├── components/            # 재사용 가능한 컴포넌트
│   ├── Chat.jsx           # 채팅 아이템 컴포넌트
│   ├── Chatlist.jsx       # 채팅 목록
│   ├── ChatMessages.jsx   # 채팅 메시지
│   └── Sidebar.jsx        # 사이드바 네비게이션
├── contexts/              # React Context
│   ├── AuthContext.js     # 인증 상태 관리
│   └── ChatContext.js     # 채팅 상태 관리
└── lib/                   # 유틸리티 및 API
    ├── api.js             # API 클라이언트
    ├── chatapi.js         # 채팅 API
    └── websocket.js       # WebSocket 연결
```

## 🛠️ 설치 및 실행

### 1. 저장소 클론
```bash
git clone <repository-url>
cd optimo_fe
```

### 2. 의존성 설치
```bash
npm install

### 3. 환경 변수 설정
프로젝트 루트에 `.env.local` 파일을 생성하고 다음을 추가하세요:

```env
NEXT_PUBLIC_API_URL=
NEXT_PUBLIC_CHATAPI_URL=
```

### 4. 개발 서버 실행
```bash
npm run dev
# 또는
yarn dev
```

브라우저에서 [http://localhost:3000](http://localhost:3000)을 열어 확인하세요.

## 🎯 사용 방법

### 1. **로그인하기**
- 홈페이지에서 이메일 주소 입력
- "제출" 버튼 클릭으로 즉시 로그인

### 2. **AI와 대화하기**
- 사이드바의 채팅 아이콘 클릭
- 메시지 입력 후 전송

### 3. **탄소배출량 확인하기**
- 사이드바의 개요 아이콘 클릭
- 실시간 탄소발자국 및 아낀 전력 소비량 확인
- Net Zero 달성 진행률 확인
- 친환경 AI 사용 가이드 참고

## 🎨 디자인 시스템

### 색상 팔레트
- **Primary**: `#84cc16` (Lime-400) - 환경 친화적 그린, Net Zero 상징
- **Background**: `#1F2123` (Dark Gray) - 눈의 피로도 감소
- **Text**: `#ffffff` (White) - 높은 가독성
- **Secondary Text**: `#9ca3af` (Gray-400) - 부드러운 대비

### 폰트
- **Sans**: Geist Sans (Variable Font) - 현대적이고 깔끔한 디자인
- **Mono**: Geist Mono (Variable Font) - 코드 및 수학 공식용

### 애니메이션
- **Glow Effect**: 빛나는 점 효과로 미래지향적 느낌
- **Slide Up**: 부드러운 등장 애니메이션
- **Hover Effects**: 인터랙티브한 사용자 경험

## 📱 페이지 구조

1. **홈페이지 (`/`)**: 로그인 화면
2. **채팅 (`/chat`)**: AI 채팅 인터페이스
3. **개요 (`/overview`)**: 탄소배출량 및 Net Zero 진행률
4. **프로필 (`/profile`)**: 환경 친화 목표 및 개인 통계

## 🚀 배포

### 빌드
```bash
npm run build
```

### 프로덕션 실행
```bash
npm start
```


## 🧪 개발 가이드

### 코드 스타일
- ESLint 설정 사용
- Prettier 포맷팅 권장
- 컴포넌트는 PascalCase
- 파일명은 camelCase 또는 kebab-case

### 상태 관리
- 전역 상태: React Context API 사용
- 로컬 상태: useState, useEffect 사용
- 인증 상태: AuthContext에서 관리

### 컴포넌트 작성 규칙
- 함수형 컴포넌트 사용
- Props 타입 검증 권장
- 재사용 가능한 컴포넌트로 분리


## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.

---

**Optimo** - Net Zero를 위한 스마트한 AI 선택 🌍♻️
