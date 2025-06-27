package org.inhahackers.optimo_chatting_be.util;

import org.inhahackers.optimo_chatting_be.exception.InvalidAuthorizationHeaderException;

public class AuthorizationHeaderUtil {

    private static final String BEARER_PREFIX = "Bearer ";

    public static String extractToken(String authHeader) {
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            throw new InvalidAuthorizationHeaderException("Authorization 헤더가 없거나 형식이 잘못되었습니다.");
        }
        return authHeader.substring(7);
    }
}