package org.inhahackers.optimo_chatting_be.controller;

import org.inhahackers.optimo_chatting_be.entity.Chatting;
import org.inhahackers.optimo_chatting_be.exception.InvalidAuthorizationHeaderException;
import org.inhahackers.optimo_chatting_be.exception.JwtAuthenticationException;
import org.inhahackers.optimo_chatting_be.service.ChattingService;
import org.inhahackers.optimo_chatting_be.service.JwtTokenVerifyService;
import org.inhahackers.optimo_chatting_be.util.AuthorizationHeaderUtil;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.server.ServerResponse;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping(("/api"))
public class ChattingController {
    private final JwtTokenVerifyService jwtTokenVerifyService;
    private final ChattingService chattingService;

    public ChattingController(JwtTokenVerifyService jwtTokenVerifyService, ChattingService chattingService) {
        this.jwtTokenVerifyService = jwtTokenVerifyService;
        this.chattingService = chattingService;
    }

    @GetMapping("/chattings/not")
    public ResponseEntity<?> getChattingsByOwnerIdWithoutChatList(
            @RequestHeader(name = "Authorization") String authHeader
    ) {
        try {
            // Authorization 헤더에서 토큰 추출
            String accessToken = AuthorizationHeaderUtil.extractToken(authHeader);

            // 토큰에서 userId 추출
            Long userId = jwtTokenVerifyService.extractUserId(accessToken);

            // 서비스 호출
            Object responseBody = chattingService.findAllByOwnerIdWithoutChatList(userId.toString());

            return ResponseEntity.ok()
                    .header("Content-Type", "application/json")
                    .body(responseBody);

        } catch (InvalidAuthorizationHeaderException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body("Authorization 헤더가 잘못되었습니다: " + e.getMessage());
        } catch (JwtAuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body("인증 실패: " + e.getMessage());
        } catch (NumberFormatException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body("잘못된 사용자 ID 형식: " + e.getMessage());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("서버 내부 오류: " + e.getMessage());
        }
    }

    @GetMapping("/chattings")
    public ResponseEntity<?> getChattingsByOwnerIdWithChatList(
            @RequestHeader(name = "Authorization") String authHeader
    ) {
        try {
            String accessToken = AuthorizationHeaderUtil.extractToken(authHeader);
            Long userId = jwtTokenVerifyService.extractUserId(accessToken);

            Object chatList = chattingService.findAllByOwnerId(userId.toString());

            return ResponseEntity.ok()
                    .header("Content-Type", "application/json")
                    .body(chatList);

        } catch (InvalidAuthorizationHeaderException | NumberFormatException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .header("Content-Type", "text/plain")
                    .body("Invalid request: " + e.getMessage());

        } catch (JwtAuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .header("Content-Type", "text/plain")
                    .body("Authentication failed: " + e.getMessage());
        }
    }

    @GetMapping("/chatting/{chattingId}")
    public ResponseEntity<?> getChattingById(
            @RequestHeader(name = "Authorization") String authHeader,
            @PathVariable String chattingId) {

        String accessToken = AuthorizationHeaderUtil.extractToken(authHeader);

        try {
            Long userId = jwtTokenVerifyService.extractUserId(accessToken);

            Chatting chatting = chattingService.findById(chattingId, userId.toString());

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(chatting);

        } catch (JwtAuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .contentType(MediaType.TEXT_PLAIN)
                    .body("Authentication failed: " + e.getMessage());
        }
    }

    @DeleteMapping("/chatting/{chattingId}")
    public ResponseEntity<?> deleteChattingById(
            @RequestHeader(name = "Authorization") String authHeader,
            @PathVariable String chattingId
    ) {
        try {
            String accessToken = AuthorizationHeaderUtil.extractToken(authHeader);

            Long userId = jwtTokenVerifyService.extractUserId(accessToken);

            chattingService.deleteById(chattingId, userId.toString());

            return ResponseEntity.noContent().build();

        } catch (InvalidAuthorizationHeaderException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body("Authorization 헤더가 잘못되었습니다: " + e.getMessage());
        } catch (JwtAuthenticationException e) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body("인증 실패: " + e.getMessage());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.status(HttpStatus.NOT_FOUND)
                    .body("채팅이 존재하지 않습니다: " + e.getMessage());
        } catch (SecurityException e) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN)
                    .body("삭제 권한이 없습니다: " + e.getMessage());
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("서버 내부 오류: " + e.getMessage());
        }
    }
}
