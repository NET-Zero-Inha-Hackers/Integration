import { NextResponse } from 'next/server';

export function middleware(request) {
  const { pathname } = request.nextUrl;
  
  // 인증이 필요하지 않은 경로들 (공개 경로)
  const publicPaths = ['/auth/login', '/api/auth/logout', '/'];
  
  // 현재 경로가 공개 경로인지 확인
  const isPublicPath = publicPaths.some(path => pathname === path);
  
  // 쿠키에서 JWT 토큰 확인
  const token = request.cookies.get('jwtToken')?.value;
  
  console.log('🔍 Pathname:', pathname);
  console.log('🔍 Token exists:', !!token);
  console.log('🔍 Public path:', isPublicPath);
  
  // 토큰이 없고 공개 경로가 아닌 경우 로그인 페이지로 리다이렉트
  if (!token && !isPublicPath) {
    console.log('🔒 인증 필요 - /으로 리다이렉트');
    return NextResponse.redirect(new URL('/', request.url));
  }
  
  // 토큰이 있고 로그인 페이지에 있는 경우 메인 페이지로 리다이렉트
  if (token && pathname === '/') {
    console.log('🔓 이미 로그인됨 - /chat으로 리다이렉트');
    return NextResponse.redirect(new URL('/chat', request.url));
  }
  
  console.log('✅ 정상 접근 허용');
  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
}; 