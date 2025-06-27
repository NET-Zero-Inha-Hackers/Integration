# Optimo_Chatting_BE

Optimo 프로젝트의 Chatting 조회/삭 서비스 Backend Server용 레포지토리입니다.

---
# BE Server 테스트 및 실행

## 환경

- JDK 17, SpringBoot 3.5, Java + Gradle

## Connected Infra

- MongoDB
    - 환경 변수
        - `MONGODB_URI` : MongoDB 연결 URI

## 실행 방법
```
# Chatting API 명세서 (Markdown)

## 1. **채팅 목록 조회 (채팅 내용 미포함)**

**URL:** `/api/chattings/not`
**Method:** `GET`
**요청 헤더:**

- `Authorization: Bearer [JWT 토큰]`
**설명:**
사용자가 소유한 채팅 목록을 반환합니다. 채팅 내용(chatList)은 포함하지 않습니다.
**응답:**
- **성공 (200):**

```json
[
  {
    "chattingId": "chat1",
    "ownerId": "user123",
    "title": "첫 번째 채팅",
    "description": "테스트 채팅입니다.",
    "createdAt": 1620000000,
    "modifiedAt": 1620000001
  }
]
```

- **실패 (401):**
    - `Authorization 헤더가 잘못되었습니다: [상세]`
    - `인증 실패: [상세]`
- **실패 (500):**
    - `서버 내부 오류: [상세]`


## 2. **채팅 목록 조회 (채팅 내용 포함)**

**URL:** `/api/chattings`
**Method:** `GET`
**요청 헤더:**

- `Authorization: Bearer [JWT 토큰]`
**설명:**
사용자가 소유한 채팅 목록을 반환합니다. 채팅 내용(chatList)이 포함됩니다.
**응답:**
- **성공 (200):**

```json
[
  {
    "chattingId": "chat1",
    "ownerId": "user123",
    "title": "첫 번째 채팅",
    "description": "테스트 채팅입니다.",
    "chatList": [
      {
        "sender": "user",
        "text": "안녕하세요",
        "timestamp": 1620000000,
        "model": "gpt-4",
        "use_estimate": 100,
        "llm_estimate": 50
      }
    ],
    "createdAt": 1620000000,
    "modifiedAt": 1620000001
  }
]
```

- **실패 (401):**
    - `Invalid request: [상세]`
    - `Authentication failed: [상세]`
- **실패 (500):**
    - `Internal error: [상세]`


## 3. **특정 채팅 조회**

**URL:** `/api/chatting/{chattingId}`
**Method:** `GET`
**경로 파라미터:**

- `chattingId`: 채팅 ID (문자열)
**요청 헤더:**
- `Authorization: Bearer [JWT 토큰]`
**설명:**
특정 채팅의 상세 정보를 반환합니다.
**응답:**
- **성공 (200):**

```json
{
  "chattingId": "chat1",
  "ownerId": "user123",
  "title": "첫 번째 채팅",
  "description": "테스트 채팅입니다.",
  "chatList": [
    {
      "sender": "user",
      "text": "안녕하세요",
      "timestamp": 1620000000,
      "model": "gpt-4",
      "use_estimate": 100,
      "llm_estimate": 50
    }
  ],
  "createdAt": 1620000000,
  "modifiedAt": 1620000001
}
```

- **실패 (401):**
    - `Authentication failed: [상세]`
- **실패 (404):**
    - `채팅이 존재하지 않습니다: [상세]` (실제 구현에 따라 다를 수 있음)
- **실패 (500):**
    - `Internal error: [상세]`


## 4. **채팅 삭제**

**URL:** `/api/chatting/{chattingId}`
**Method:** `DELETE`
**경로 파라미터:**

- `chattingId`: 채팅 ID (문자열)
**요청 헤더:**
- `Authorization: Bearer [JWT 토큰]`
**설명:**
특정 채팅을 삭제합니다.
**응답:**
- **성공 (204):**
    - 본문 없음 (No Content)
- **실패 (401):**
    - `Authorization 헤더가 잘못되었습니다: [상세]`
    - `인증 실패: [상세]`
- **실패 (403):**
    - `삭제 권한이 없습니다: [상세]`
- **실패 (404):**
    - `채팅이 존재하지 않습니다: [상세]`
- **실패 (500):**
    - `서버 내부 오류: [상세]`


## 5. **채팅 생성 (예시, 실제 구현에 따라 다름)**

**URL:** `/api/chatting`
**Method:** `POST`
**요청 헤더:**

- `Authorization: Bearer [JWT 토큰]`
**요청 본문:**

```json
{
  "title": "새 채팅",
  "description": "새로운 채팅입니다."
}
```

**응답:**

- **성공 (201):**

```json
{
  "chattingId": "chat2",
  "ownerId": "user123",
  "title": "새 채팅",
  "description": "새로운 채팅입니다.",
  "chatList": [],
  "createdAt": 1620000002,
  "modifiedAt": 1620000002
}
```

- **실패 (400):**
    - `잘못된 요청: [상세]`
- **실패 (401):**
    - `인증 실패: [상세]`
- **실패 (500):**
    - `서버 내부 오류: [상세]`


## 6. **채팅 메시지 추가 (예시, 실제 구현에 따라 다름)**

**URL:** `/api/chatting/{chattingId}/message`
**Method:** `POST`
**경로 파라미터:**

- `chattingId`: 채팅 ID (문자열)
**요청 헤더:**
- `Authorization: Bearer [JWT 토큰]`
**요청 본문:**

```json
{
  "sender": "user",
  "text": "새 메시지",
  "model": "gpt-4",
  "use_estimate": 100,
  "llm_estimate": 50
}
```

**응답:**

- **성공 (200):**

```json
{
  "chattingId": "chat1",
  "ownerId": "user123",
  "title": "첫 번째 채팅",
  "description": "테스트 채팅입니다.",
  "chatList": [
    {
      "sender": "user",
      "text": "안녕하세요",
      "timestamp": 1620000000,
      "model": "gpt-4",
      "use_estimate": 100,
      "llm_estimate": 50
    },
    {
      "sender": "user",
      "text": "새 메시지",
      "timestamp": 1620000003,
      "model": "gpt-4",
      "use_estimate": 100,
      "llm_estimate": 50
    }
  ],
  "createdAt": 1620000000,
  "modifiedAt": 1620000003
}
```

- **실패 (400):**
    - `잘못된 요청: [상세]`
- **실패 (401):**
    - `인증 실패: [상세]`
- **실패 (404):**
    - `채팅이 존재하지 않습니다: [상세]`
- **실패 (500):**
    - `서버 내부 오류: [상세]`


## **공통 사항**

- **인증:** 모든 API는 `Authorization: Bearer [JWT 토큰]` 헤더가 필요합니다.
- **에러 처리:**
    - `401 Unauthorized`: 인증 실패
    - `403 Forbidden`: 권한 없음
    - `404 Not Found`: 리소스 없음
    - `500 Internal Server Error`: 서버 내부 오류


## **요약 표**

| 엔드포인트 | 메소드 | 설명 | 성공 응답 | 실패 응답 |
| :-- | :-- | :-- | :-- | :-- |
| `/api/chattings/not` | GET | 채팅 목록(내용 미포함) | 200 | 401, 500 |
| `/api/chattings` | GET | 채팅 목록(내용 포함) | 200 | 401, 500 |
| `/api/chatting/{chattingId}` | GET | 특정 채팅 조회 | 200 | 401, 404, 500 |
| `/api/chatting/{chattingId}` | DELETE | 채팅 삭제 | 204 | 401, 403, 404, 500 |

> **참고:**
> - **실제 구현에 따라 엔드포인트, 요청/응답 형식, 상태 코드가 다를 수 있습니다.**
> - **채팅 생성/메시지 추가는 예시이며, 구현이 되어 있지 않다면 생략해도 무방합니다.**
> - **JWT 토큰은 모든 API에서 필수입니다.**
# 이전 빌드 제거 및 빌드 명령어
./gradlew clean bootJar

# 빌드된 jar 파일 실행 명령어
java -jar ./build/libs/Optimo_User_BE-0.0.1-SNAPSHOT.jar
